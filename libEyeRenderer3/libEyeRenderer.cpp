//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "libEyeRenderer.h"

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "MulticamScene.h"
#include "GlobalParameters.h"
#include "cameras/CompoundEyeDataTypes.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include <vector>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards
#ifdef BUFFER_TYPE_CUDA_DEVICE
  #define BUFFER_TYPE 0
#endif
#ifdef BUFFER_TYPE_GL_INTEROP
  #define BUFFER_TYPE 1
#endif
#ifdef BUFFER_TYPE_ZERO_COPY
  #define BUFFER_TYPE 2
#endif
#ifdef BUFFER_TYPE_CUDA_P2P
  #define BUFFER_TYPE 3
#endif

MulticamScene scene;

globalParameters::LaunchParams*  d_params = nullptr;
globalParameters::LaunchParams   params   = {};
int32_t                 width    = 400;
int32_t                 height   = 400;

GLFWwindow* window = sutil::initUI( "Eye Renderer 3.0", width, height );
sutil::CUDAOutputBuffer<uchar4> outputBuffer(static_cast<sutil::CUDAOutputBufferType>(BUFFER_TYPE), width, height);
sutil::GLDisplay gl_display; // Stores the frame buffer to swap in and out

bool notificationsActive = true;

//------------------------------------------------------------------------------
// Light setup
//------------------------------------------------------------------------------
//
// History / why this code looks the way it does:
//
// Before these changes the renderer didnt read lights although 
// there were dependencies added in. The renderer would only read
// base color of the textures, i.e, what a material preview in Blender
// does. I honestly wouldve spent less time just devloping my own tool
// than i did trying to fix all of this stuff.
//
// For the most parts the lights are hardcoded in although there is
// now dependency to add a light from a GLTF
// 
// The previous version positioned four point lights at
//     scene.aabb().center() + make_float3( +/- maxExtent, ... )
// where loffset == aabb.maxExtent(). For scenes whose AABB is dominated
// by an axis-aligned enclosing shell this
// places lights at the *corners* of the AABB -- always outside the
// inscribed shell. Every shadow ray then has to pass through the shell
// wall on its way to the light, so every light reads as occluded and
// the entire scene renders pure black.
//
// We now do two things differently:
//   1. Try to pull lights from the loaded glTF scene first
//      (KHR_lights_punctual). This honours whatever the artist set up
//      in Blender -- e.g. the "Sun" directional light in the test
//      scene. Requires MulticamScene::getPunctualLights() to be
//      implemented; if it returns an empty list we fall through.
//   2. Fall back to a set of synthetic point lights placed on a
//      shrunken offset (~25% of the AABB diagonal) so they end up
//      well inside any reasonable enclosing shell.
//
// Shadow rays themselves now offset their origin along the geometric
// normal, and inverted-shell geometry should be bound to the
// __anyhit__occlusion_passthrough program in the SBT so it cannot
// occlude shadow rays at all. See shaders.cu and NEXT_STEPS.md.
//------------------------------------------------------------------------------

namespace {

// Convert a glTF directional light into a far-away point light we can plug into the existing
// Light::Point pipeline. This is a pragmatic stop-gap until the shader
// gains a proper directional-light type. The "far away" distance is
// chosen relative to the scene's AABB so the light stays effectively
// directional regardless of scene scale.
Light::Point makeDirectionalAsPoint(
        const float3& direction_to_light,
        const float3& color,
        float intensity,
        const sutil::Aabb& aabb )
{
    const float far_dist = 10.0f * length( aabb.extent() );
    Light::Point l;
    l.position  = aabb.center() + normalize( direction_to_light ) * far_dist;
    l.color     = color;
    // Account for the inverse-square falloff over far_dist so a directional
    // light "feels" the same intensity as the artist intended. The shader
    // applies QUADRATIC falloff implicitly via the standard rendering
    // equation, so we pre-multiply here.
    l.intensity = intensity * (far_dist * far_dist);
    l.falloff   = Light::Falloff::QUADRATIC;
    return l;
}

// Build the synthetic-light fallback set. Lights are placed on a small
// offset relative to the scene's interior, NOT at the AABB corners,
// so they stay inside enclosing-shell geometry.
std::vector<Light::Point> buildFallbackLights( const sutil::Aabb& aabb )
{
    const float3 center  = aabb.center();
    const float3 extent  = aabb.extent();

    // Use a fraction of the *diagonal* rather than the maximum extent.
    // 0.25 of the diagonal puts the light comfortably inside any axis-
    // aligned cylinder or sphere whose radius is at least 0.5 * smallest
    // extent, which covers virtually every artist-built sky-cage.
    const float r = 0.25f * length( extent );

    std::vector<Light::Point> lights(4);
    lights[0].color     = { 1.0f, 1.0f, 0.8f };
    lights[0].intensity = 5;  // scale so apparent brightness is roughly
                                         // independent of scene scale
    lights[0].position  =  make_float3( 0.0f,100.0f,0.0f);
    lights[0].falloff   = Light::Falloff::QUADRATIC;

    lights[1].color     = { 0.8f, 0.8f, 1.0f };
    lights[1].intensity = 0.0f * r * r;
    lights[1].position  = make_float3(0.0f,0.0f,0.0f);
    lights[1].falloff   = Light::Falloff::QUADRATIC;

    lights[2].color     = { 1.0f, 1.0f, 0.8f };
    lights[2].intensity = 0.0f * r * r;
    lights[2].position  = center + make_float3( 0.0f,  0.5f*r, -0.7f*r );
    lights[2].falloff   = Light::Falloff::QUADRATIC;

    lights[3].color     = { 1.0f, 1.0f, 0.8f };
    lights[3].intensity = 0.0f * r * r;
    lights[3].position  = center + make_float3( 0.2f*r, -0.7f*r, 0.0f );
    lights[3].falloff   = Light::Falloff::QUADRATIC;

    return lights;
}

// Pull lights from the scene if MulticamScene exposes them. We use a
// SFINAE-style guard (preprocessor) so this file still compiles cleanly
// against an older MulticamScene that doesn't have getPunctualLights().
// Once MulticamScene is updated (see NEXT_STEPS.md), define
// MULTICAM_SCENE_HAS_PUNCTUAL_LIGHTS and the branch lights up.
std::vector<Light::Point> buildLightsFromScene( const MulticamScene& s )
{
    std::vector<Light::Point> lights;
#ifdef MULTICAM_SCENE_HAS_PUNCTUAL_LIGHTS
    const auto& gltf_lights = s.getPunctualLights(); // see NEXT_STEPS.md for the type
    lights.reserve( gltf_lights.size() );
    for( const auto& gl : gltf_lights )
    {
        if( gl.type == GltfLight::DIRECTIONAL )
        {
            lights.push_back(
                makeDirectionalAsPoint( gl.direction_to_light, gl.color, gl.intensity, s.aabb() ) );
        }
        else // POINT or SPOT (treat spot as point for now)
        {
            Light::Point p;
            p.color     = gl.color;
            p.intensity = gl.intensity;
            p.position  = gl.position;
            p.falloff   = Light::Falloff::QUADRATIC;
            lights.push_back( p );
        }
    }
#else
    (void)s;
#endif
    return lights;
}

} // anonymous namespace

void initLaunchParams( const MulticamScene& scene ) {

    params.frame_buffer = nullptr; // Will be set when output buffer is mapped
    params.frame = 0;
    params.lighting = true;

    // 1) Prefer artist-defined lights from the glTF.
    std::vector<Light::Point> lights = buildLightsFromScene( scene );

    // 2) Fall back to interior synthetic lights if no glTF lights were found.
    if( lights.empty() )
    {
        if( notificationsActive )
            std::cout << "[PyEye] No punctual lights found in scene, using fallback set." << std::endl;
        lights = buildFallbackLights( scene.aabb() );
    }
    else if( notificationsActive )
    {
        std::cout << "[PyEye] Loaded " << lights.size() << " light(s) from glTF." << std::endl;
    }

    params.lights.count  = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.lights.data ),
                lights.size() * sizeof( Light::Point )
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( params.lights.data ),
                lights.data(),
                lights.size() * sizeof( Light::Point ),
                cudaMemcpyHostToDevice
                ) );

    params.miss_color   = make_float3( 0.1f );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( globalParameters::LaunchParams ) ) );

    params.handle = scene.traversableHandle();
}


// Updates the params to acurately reflect the currently selected camera
void handleCameraUpdate( globalParameters::LaunchParams& params )
{
    // Make sure the SBT of the scene is updated for the newly selected camera before launch,
    // also push any changed host-side camera SBT data over to the device.
    scene.reconfigureSBTforCurrentCamera(false);
}


void launchFrame( sutil::CUDAOutputBuffer<uchar4>& output_buffer, MulticamScene& scene )
{
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer        = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
                &params,
                sizeof( globalParameters::LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    if(scene.hasCompoundEyes() && scene.isCompoundEyeActive())
    {
      CompoundEye* camera = (CompoundEye*) scene.getCamera();
      // Launch the ommatidial renderer
      OPTIX_CHECK( optixLaunch(
                  scene.compoundPipeline(),
                  0,             // stream
                  reinterpret_cast<CUdeviceptr>( d_params ),
                  sizeof( globalParameters::LaunchParams ),
                  scene.compoundSbt(),
                  camera->getOmmatidialCount(),      // launch width
                  camera->getSamplesPerOmmatidium(), // launch height
                  1                                  // launch depth
                  ) );
      CUDA_SYNC_CHECK();
      params.frame++;// Increase the frame number
      camera->setRandomsAsConfigured();// Make sure that random stream initialization is only ever done once
    }

    // Launch render
    OPTIX_CHECK( optixLaunch(
                scene.pipeline(),
                0,             // stream
                reinterpret_cast<CUdeviceptr>( d_params ),
                sizeof( globalParameters::LaunchParams ),
                scene.sbt(),
                width,  // launch width
                height, // launch height
                1
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void cleanup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params               ) ) );
    scene.cleanup();
}

//------------------------------------------------------------------------------
//
// API functions
//
//------------------------------------------------------------------------------
// General Running
//------------------------------------------------------------------------------
void setVerbosity(bool v)
{
  notificationsActive = v;
}
void loadGlTFscene(const char* filepath)
{
  loadScene(filepath, scene);
  scene.finalize();
  initLaunchParams(scene);
}
void setRenderSize(int w, int h)
{
  width = w;
  height = h;
  if(notificationsActive)
    std::cout<<"[PyEye] Resizing rendering buffer to ("<<w<<", "<<h<<")."<<std::endl;
  outputBuffer.resize(width, height);
}
double renderFrame(void)
{
  handleCameraUpdate(params);// Update the params to accurately reflect the currently selected camera

  auto then = std::chrono::steady_clock::now();
  launchFrame( outputBuffer, scene );
  std::chrono::duration<double, std::milli> render_time = std::chrono::steady_clock::now() - then;

  if(notificationsActive)
    std::cout<<"[PyEye] Rendered frame in "<<render_time.count()<<"ms."<<std::endl;

  CUDA_SYNC_CHECK();
  return(render_time.count());
}
void displayFrame(void)
{
  int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
  int framebuf_res_y = 0;   //
  glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
  gl_display.display(
          outputBuffer.width(),
          outputBuffer.height(),
          framebuf_res_x,
          framebuf_res_y,
          outputBuffer.getPBO()
          );

  // Swap the buffer
  glfwSwapBuffers(window);
}
void saveFrameAs(char* ppmFilename)
{
  sutil::ImageBuffer buffer;
  buffer.data = outputBuffer.getHostPointer();
  buffer.width = outputBuffer.width();
  buffer.height = outputBuffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
  sutil::displayBufferFile(ppmFilename, buffer, false);
  if(notificationsActive)
    std::cout<<"[PyEye] Saved render as '"<<ppmFilename<<"'"<<std::endl;
}
unsigned char* getFramePointer(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame pointer..."<<std::endl;
  return (unsigned char*)outputBuffer.getHostPointer();
}
void getFrame(unsigned char* frame)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Retrieving frame..."<<std::endl;
  size_t displaySize = outputBuffer.width()*outputBuffer.height();
  for(size_t i = 0; i<displaySize; i++)
  {
    unsigned char val = (unsigned char)(((float)i/(float)displaySize)*254);
    frame[displaySize*3 + 0] = val;
    frame[displaySize*3 + 1] = val;
    frame[displaySize*3 + 2] = val;
  }
}
void stop(void)
{
  if(notificationsActive)
    std::cout<<"[PyEye] Cleaning eye renderer resources."<<std::endl;
  sutil::cleanupUI(window);
  cleanup();
}

// C-level only
void * getWindowPointer()
{
  return (void*)window;
}

//------------------------------------------------------------------------------
// Camera Control
//------------------------------------------------------------------------------
size_t getCameraCount()
{
  return(scene.getCameraCount());
}
void nextCamera(void)
{
  scene.nextCamera();
}
size_t getCurrentCameraIndex(void)
{
  return(scene.getCameraIndex());
}
const char* getCurrentCameraName(void)
{
  return(scene.getCamera()->getCameraName());
}
void previousCamera(void)
{
  scene.previousCamera();
}
void gotoCamera(int index)
{
  scene.setCurrentCamera(index);
}
bool gotoCameraByName(char* name)
{
  scene.setCurrentCamera(0);
  for(auto i = 0; i<scene.getCameraCount(); i++)
  {
    if(strcmp(name, scene.getCamera()->getCameraName()) == 0)
      return true;
    scene.nextCamera();
  }
  return false;
}
void setCameraPosition(float x, float y, float z)
{
  scene.getCamera()->setPosition(make_float3(x,y,z));
}
void getCameraPosition(float& x, float& y, float& z)
{
  const float3& camPos = scene.getCamera()->getPosition();
  x = camPos.x;
  y = camPos.y;
  z = camPos.z;
}
void setCameraLocalSpace(float lxx, float lxy, float lxz,
                         float lyx, float lyy, float lyz,
                         float lzx, float lzy, float lzz)
{
  scene.getCamera()->setLocalSpace(make_float3(lxx, lxy, lxz),
                                   make_float3(lyx, lyy, lyz),
                                   make_float3(lzx, lzy, lzz));
}
void rotateCameraAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateAround(angle,  make_float3(x,y,z));
}
void rotateCameraLocallyAround(float angle, float x, float y, float z)
{
  scene.getCamera()->rotateLocallyAround(angle,  make_float3(x,y,z));
}
void translateCamera(float x, float y, float z)
{
  scene.getCamera()->move(make_float3(x, y, z));
}
void translateCameraLocally(float x, float y, float z)
{
  scene.getCamera()->moveLocally(make_float3(x, y, z));
}
void resetCameraPose()
{
  scene.getCamera()->resetPose();
}
void setCameraPose(float posX, float posY, float posZ, float rotX, float rotY, float rotZ)
{
  GenericCamera* c = scene.getCamera();
  c->resetPose();
  c->rotateAround(rotX, make_float3(1,0,0));
  c->rotateAround(rotY, make_float3(0,1,0));
  c->rotateAround(rotZ, make_float3(0,0,1));
  c->move(make_float3(posX, posY, posZ));
}

//------------------------------------------------------------------------------
// Ommatidial Camera Control
//------------------------------------------------------------------------------
bool isCompoundEyeActive(void)
{
  return scene.isCompoundEyeActive();
}
void setCurrentEyeSamplesPerOmmatidium(int s)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->setSamplesPerOmmatidium(s);
  }
}
int getCurrentEyeSamplesPerOmmatidium(void)
{
  if(scene.isCompoundEyeActive())
  {
    return(((CompoundEye*)scene.getCamera())->getSamplesPerOmmatidium());
  }
  return -1;
}
void changeCurrentEyeSamplesPerOmmatidiumBy(int s)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->changeSamplesPerOmmatidiumBy(s);
  }
}
size_t getCurrentEyeOmmatidialCount(void)
{
  if(scene.isCompoundEyeActive())
  {
    return ((CompoundEye*)scene.getCamera())->getOmmatidialCount();
  }
  return 0;
}
void setOmmatidia(OmmatidiumPacket* omms, size_t count)
{
  if(!scene.isCompoundEyeActive())
    return;

  std::vector<Ommatidium> ommVector(count);
  for(size_t i = 0; i<count; i++)
  {
    OmmatidiumPacket& omm = omms[i];
    ommVector[i].relativePosition  = make_float3(omm.posX, omm.posY, omm.posZ);
    ommVector[i].relativeDirection = make_float3(omm.dirX, omm.dirY, omm.dirZ);
    ommVector[i].acceptanceAngleRadians = omm.acceptanceAngle;
    ommVector[i].focalPointOffset = omm.focalpointOffset;
  }

  ((CompoundEye*)scene.getCamera())->setOmmatidia(ommVector.data(), count);
}
const char* getCurrentEyeDataPath(void)
{
  if(scene.isCompoundEyeActive())
  {
    return ((CompoundEye*)scene.getCamera())->eyeDataPath.c_str();
  }
  return "\0";
}
void setCurrentEyeShaderName(char* name)
{
  if(scene.isCompoundEyeActive())
  {
    ((CompoundEye*)scene.getCamera())->setShaderName(std::string(name)); // Set the shader
    scene.reconfigureSBTforCurrentCamera(true); // Reconfigure for the new shader
  }
}

bool isInsideHitGeometry(float x, float y, float z, char* name)
{
  return scene.isInsideHitGeometry(make_float3(x, y, z), std::string(name), false);
}
float3 getGeometryMaxBounds(char* name)
{
  return scene.getGeometryMaxBounds(std::string(name));
}
float3 getGeometryMinBounds(char* name)
{
  return scene.getGeometryMinBounds(std::string(name));
}
