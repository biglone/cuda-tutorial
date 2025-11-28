/**
 * =============================================================================
 * CUDA 教程 21: CUDA 与图形 API 互操作 (Graphics Interop)
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 CUDA 与 OpenGL 互操作的基本概念
 * 2. 学会注册和映射 OpenGL 资源
 * 3. 掌握 PBO (Pixel Buffer Object) 和 VBO 的 CUDA 访问
 * 4. 了解与 Vulkan 互操作的基本原理
 *
 * 关键概念：
 * - Graphics Interop: 在 CUDA 和图形 API 之间共享内存
 * - 零拷贝: 避免 CPU-GPU 数据传输
 * - 实时渲染: GPU 计算结果直接用于渲染
 *
 * 编译命令 (需要 OpenGL 库):
 *   nvcc 21_interop_graphics.cu -o 21_interop_graphics -lGL -lGLU -lglut
 *
 * 注意: 此教程主要演示概念和 API 用法
 *       完整运行需要 OpenGL 环境和显示设备
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：图形互操作概念介绍
// ============================================================================

void demoInteropConcepts() {
    printf("=== 第一部分：图形互操作概念 ===\n\n");

    printf("什么是图形互操作 (Graphics Interop)？\n");
    printf("  在 CUDA 和图形 API (OpenGL/Vulkan) 之间共享 GPU 内存\n");
    printf("  避免数据在 CPU 和 GPU 之间来回复制\n\n");

    printf("主要应用场景:\n");
    printf("  1. 实时图像/视频处理 + 显示\n");
    printf("  2. 物理模拟 + 可视化\n");
    printf("  3. 粒子系统渲染\n");
    printf("  4. 科学可视化\n");
    printf("  5. VR/AR 应用\n\n");

    printf("支持的图形 API:\n");
    printf("  - OpenGL (最常用，跨平台)\n");
    printf("  - Direct3D 9/10/11/12 (Windows)\n");
    printf("  - Vulkan (新一代跨平台 API)\n\n");

    printf("工作流程:\n");
    printf("  ┌─────────────────────────────────────────┐\n");
    printf("  │  1. 创建图形 API 资源 (Buffer/Texture)  │\n");
    printf("  │             ↓                           │\n");
    printf("  │  2. 注册资源到 CUDA                     │\n");
    printf("  │             ↓                           │\n");
    printf("  │  3. 映射资源获取 CUDA 指针              │\n");
    printf("  │             ↓                           │\n");
    printf("  │  4. CUDA 内核处理数据                   │\n");
    printf("  │             ↓                           │\n");
    printf("  │  5. 解除映射                            │\n");
    printf("  │             ↓                           │\n");
    printf("  │  6. 图形 API 使用资源渲染               │\n");
    printf("  └─────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：OpenGL 互操作 API 概览
// ============================================================================

void demoOpenGLInteropAPI() {
    printf("=== 第二部分：OpenGL 互操作 API ===\n\n");

    printf("核心 API 函数:\n\n");

    printf("1. 设置 CUDA-OpenGL 设备:\n");
    printf("   cudaGLSetGLDevice(device)\n");
    printf("   - 在使用互操作前必须调用\n");
    printf("   - 指定要与 OpenGL 共享的 CUDA 设备\n\n");

    printf("2. 注册 OpenGL 缓冲区:\n");
    printf("   cudaGraphicsGLRegisterBuffer(\n");
    printf("       cudaGraphicsResource **resource,\n");
    printf("       GLuint buffer,\n");
    printf("       unsigned int flags\n");
    printf("   )\n");
    printf("   flags:\n");
    printf("     - cudaGraphicsRegisterFlagsNone: 读写\n");
    printf("     - cudaGraphicsRegisterFlagsReadOnly: 只读\n");
    printf("     - cudaGraphicsRegisterFlagsWriteDiscard: 只写\n\n");

    printf("3. 注册 OpenGL 纹理:\n");
    printf("   cudaGraphicsGLRegisterImage(\n");
    printf("       cudaGraphicsResource **resource,\n");
    printf("       GLuint image,\n");
    printf("       GLenum target,\n");
    printf("       unsigned int flags\n");
    printf("   )\n");
    printf("   target: GL_TEXTURE_2D, GL_TEXTURE_3D 等\n\n");

    printf("4. 映射资源:\n");
    printf("   cudaGraphicsMapResources(\n");
    printf("       int count,\n");
    printf("       cudaGraphicsResource **resources,\n");
    printf("       cudaStream_t stream\n");
    printf("   )\n\n");

    printf("5. 获取映射指针:\n");
    printf("   cudaGraphicsResourceGetMappedPointer(\n");
    printf("       void **devPtr,\n");
    printf("       size_t *size,\n");
    printf("       cudaGraphicsResource resource\n");
    printf("   )\n\n");

    printf("6. 解除映射:\n");
    printf("   cudaGraphicsUnmapResources(\n");
    printf("       int count,\n");
    printf("       cudaGraphicsResource **resources,\n");
    printf("       cudaStream_t stream\n");
    printf("   )\n\n");

    printf("7. 注销资源:\n");
    printf("   cudaGraphicsUnregisterResource(\n");
    printf("       cudaGraphicsResource resource\n");
    printf("   )\n\n");
}

// ============================================================================
// 第三部分：PBO (Pixel Buffer Object) 互操作示例代码
// ============================================================================

void demoPBOInteropCode() {
    printf("=== 第三部分：PBO 互操作示例代码 ===\n\n");

    printf("// PBO 互操作完整示例（伪代码）\n\n");

    printf("// === 初始化阶段 ===\n");
    printf("GLuint pbo;  // OpenGL PBO\n");
    printf("cudaGraphicsResource *cudaResource;\n\n");

    printf("// 1. 创建 OpenGL PBO\n");
    printf("glGenBuffers(1, &pbo);\n");
    printf("glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);\n");
    printf("glBufferData(GL_PIXEL_UNPACK_BUFFER, \n");
    printf("             width * height * 4,  // RGBA\n");
    printf("             NULL,\n");
    printf("             GL_DYNAMIC_DRAW);\n");
    printf("glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);\n\n");

    printf("// 2. 注册 PBO 到 CUDA\n");
    printf("cudaGraphicsGLRegisterBuffer(\n");
    printf("    &cudaResource, \n");
    printf("    pbo,\n");
    printf("    cudaGraphicsRegisterFlagsWriteDiscard\n");
    printf(");\n\n");

    printf("// === 每帧渲染 ===\n");
    printf("// 3. 映射资源\n");
    printf("cudaGraphicsMapResources(1, &cudaResource, 0);\n\n");

    printf("// 4. 获取设备指针\n");
    printf("uchar4 *d_output;\n");
    printf("size_t size;\n");
    printf("cudaGraphicsResourceGetMappedPointer(\n");
    printf("    (void**)&d_output, &size, cudaResource\n");
    printf(");\n\n");

    printf("// 5. CUDA 内核处理\n");
    printf("renderKernel<<<blocks, threads>>>(d_output, width, height);\n\n");

    printf("// 6. 解除映射\n");
    printf("cudaGraphicsUnmapResources(1, &cudaResource, 0);\n\n");

    printf("// 7. OpenGL 使用 PBO 更新纹理\n");
    printf("glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);\n");
    printf("glBindTexture(GL_TEXTURE_2D, texture);\n");
    printf("glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,\n");
    printf("                width, height, GL_RGBA,\n");
    printf("                GL_UNSIGNED_BYTE, 0);\n");
    printf("glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);\n\n");

    printf("// === 清理阶段 ===\n");
    printf("cudaGraphicsUnregisterResource(cudaResource);\n");
    printf("glDeleteBuffers(1, &pbo);\n\n");
}

// ============================================================================
// 第四部分：VBO 粒子系统示例
// ============================================================================

// 粒子结构
struct Particle {
    float3 position;
    float3 velocity;
    float4 color;
};

// 粒子更新内核
__global__ void updateParticles(Particle *particles, int numParticles,
                                 float dt, float time) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];

    // 简单的物理模拟
    // 重力
    p.velocity.y -= 9.8f * dt;

    // 更新位置
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;

    // 边界反弹
    if (p.position.y < -1.0f) {
        p.position.y = -1.0f;
        p.velocity.y = -p.velocity.y * 0.8f;  // 能量损失
    }

    // 颜色随速度变化
    float speed = sqrtf(p.velocity.x * p.velocity.x +
                        p.velocity.y * p.velocity.y +
                        p.velocity.z * p.velocity.z);
    p.color.x = fminf(1.0f, speed * 0.2f);  // R
    p.color.y = fminf(1.0f, 0.5f);          // G
    p.color.z = fmaxf(0.0f, 1.0f - speed * 0.1f);  // B
    p.color.w = 1.0f;  // A

    particles[idx] = p;
}

void demoVBOParticleSystem() {
    printf("=== 第四部分：VBO 粒子系统 ===\n\n");

    printf("粒子系统互操作示例:\n\n");

    printf("// 粒子结构\n");
    printf("struct Particle {\n");
    printf("    float3 position;  // 位置 (x, y, z)\n");
    printf("    float3 velocity;  // 速度\n");
    printf("    float4 color;     // 颜色 (RGBA)\n");
    printf("};\n\n");

    printf("// === 初始化 ===\n");
    printf("const int NUM_PARTICLES = 100000;\n");
    printf("GLuint vbo;\n");
    printf("cudaGraphicsResource *vboResource;\n\n");

    printf("// 创建 VBO\n");
    printf("glGenBuffers(1, &vbo);\n");
    printf("glBindBuffer(GL_ARRAY_BUFFER, vbo);\n");
    printf("glBufferData(GL_ARRAY_BUFFER,\n");
    printf("             NUM_PARTICLES * sizeof(Particle),\n");
    printf("             NULL, GL_DYNAMIC_DRAW);\n\n");

    printf("// 注册到 CUDA\n");
    printf("cudaGraphicsGLRegisterBuffer(&vboResource, vbo,\n");
    printf("    cudaGraphicsRegisterFlagsNone);\n\n");

    printf("// === 每帧更新 ===\n");
    printf("void updateFrame(float dt) {\n");
    printf("    // 映射\n");
    printf("    cudaGraphicsMapResources(1, &vboResource, 0);\n");
    printf("    \n");
    printf("    Particle *d_particles;\n");
    printf("    size_t size;\n");
    printf("    cudaGraphicsResourceGetMappedPointer(\n");
    printf("        (void**)&d_particles, &size, vboResource);\n");
    printf("    \n");
    printf("    // CUDA 更新粒子\n");
    printf("    int blockSize = 256;\n");
    printf("    int gridSize = (NUM_PARTICLES + blockSize - 1) / blockSize;\n");
    printf("    updateParticles<<<gridSize, blockSize>>>(\n");
    printf("        d_particles, NUM_PARTICLES, dt, totalTime);\n");
    printf("    \n");
    printf("    // 解除映射\n");
    printf("    cudaGraphicsUnmapResources(1, &vboResource, 0);\n");
    printf("}\n\n");

    printf("// === 渲染 ===\n");
    printf("void render() {\n");
    printf("    glBindBuffer(GL_ARRAY_BUFFER, vbo);\n");
    printf("    \n");
    printf("    // 设置顶点属性\n");
    printf("    glEnableVertexAttribArray(0);  // position\n");
    printf("    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,\n");
    printf("        sizeof(Particle), (void*)0);\n");
    printf("    \n");
    printf("    glEnableVertexAttribArray(1);  // color\n");
    printf("    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,\n");
    printf("        sizeof(Particle), (void*)(6*sizeof(float)));\n");
    printf("    \n");
    printf("    // 绘制点\n");
    printf("    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);\n");
    printf("}\n\n");

    // 演示内核逻辑
    printf("粒子更新内核已定义，包含:\n");
    printf("  - 重力模拟\n");
    printf("  - 位置更新\n");
    printf("  - 边界反弹\n");
    printf("  - 基于速度的颜色变化\n\n");
}

// ============================================================================
// 第五部分：纹理互操作
// ============================================================================

// 图像处理内核示例
__global__ void imageProcessingKernel(cudaSurfaceObject_t surface,
                                       int width, int height, float time) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    // 生成动态图案
    float u = (float)x / width;
    float v = (float)y / height;

    // 波纹效果
    float cx = 0.5f, cy = 0.5f;
    float dist = sqrtf((u - cx) * (u - cx) + (v - cy) * (v - cy));
    float wave = sinf(dist * 30.0f - time * 5.0f) * 0.5f + 0.5f;

    // 颜色
    uchar4 pixel;
    pixel.x = (unsigned char)(wave * 255);
    pixel.y = (unsigned char)((1.0f - dist) * 255);
    pixel.z = (unsigned char)(sinf(time) * 0.5f + 0.5f) * 255;
    pixel.w = 255;

    // 写入表面
    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

void demoTextureInterop() {
    printf("=== 第五部分：纹理互操作 ===\n\n");

    printf("OpenGL 纹理与 CUDA 互操作:\n\n");

    printf("// === 初始化 ===\n");
    printf("GLuint texture;\n");
    printf("cudaGraphicsResource *textureResource;\n");
    printf("cudaArray *cudaTextureArray;\n\n");

    printf("// 创建 OpenGL 纹理\n");
    printf("glGenTextures(1, &texture);\n");
    printf("glBindTexture(GL_TEXTURE_2D, texture);\n");
    printf("glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,\n");
    printf("             width, height, 0, GL_RGBA,\n");
    printf("             GL_UNSIGNED_BYTE, NULL);\n");
    printf("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);\n");
    printf("glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);\n\n");

    printf("// 注册纹理到 CUDA\n");
    printf("cudaGraphicsGLRegisterImage(\n");
    printf("    &textureResource,\n");
    printf("    texture,\n");
    printf("    GL_TEXTURE_2D,\n");
    printf("    cudaGraphicsRegisterFlagsSurfaceLoadStore  // 允许读写\n");
    printf(");\n\n");

    printf("// === 每帧更新 ===\n");
    printf("// 映射\n");
    printf("cudaGraphicsMapResources(1, &textureResource, 0);\n\n");

    printf("// 获取 CUDA Array\n");
    printf("cudaGraphicsSubResourceGetMappedArray(\n");
    printf("    &cudaTextureArray, textureResource, 0, 0);\n\n");

    printf("// 创建 Surface Object\n");
    printf("cudaResourceDesc resDesc = {};\n");
    printf("resDesc.resType = cudaResourceTypeArray;\n");
    printf("resDesc.res.array.array = cudaTextureArray;\n\n");

    printf("cudaSurfaceObject_t surface;\n");
    printf("cudaCreateSurfaceObject(&surface, &resDesc);\n\n");

    printf("// 启动内核写入纹理\n");
    printf("dim3 blockSize(16, 16);\n");
    printf("dim3 gridSize((width + 15)/16, (height + 15)/16);\n");
    printf("imageProcessingKernel<<<gridSize, blockSize>>>(\n");
    printf("    surface, width, height, time);\n\n");

    printf("// 清理 Surface Object\n");
    printf("cudaDestroySurfaceObject(surface);\n\n");

    printf("// 解除映射\n");
    printf("cudaGraphicsUnmapResources(1, &textureResource, 0);\n\n");

    printf("// === 渲染 ===\n");
    printf("// 纹理可直接用于 OpenGL 渲染\n");
    printf("glBindTexture(GL_TEXTURE_2D, texture);\n");
    printf("drawQuad();  // 绘制带纹理的四边形\n\n");
}

// ============================================================================
// 第六部分：Vulkan 互操作简介
// ============================================================================

void demoVulkanInterop() {
    printf("=== 第六部分：Vulkan 互操作 ===\n\n");

    printf("Vulkan 与 CUDA 互操作（CUDA 10.0+）:\n\n");

    printf("主要区别:\n");
    printf("  - Vulkan 使用外部内存/信号量扩展\n");
    printf("  - 更底层的控制\n");
    printf("  - 更好的性能潜力\n\n");

    printf("关键步骤:\n\n");

    printf("1. 导出 Vulkan 内存句柄:\n");
    printf("   VkExternalMemoryBufferCreateInfo extInfo = {\n");
    printf("       .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,\n");
    printf("       .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT\n");
    printf("   };\n\n");

    printf("2. 获取内存文件描述符:\n");
    printf("   VkMemoryGetFdInfoKHR getFdInfo = {\n");
    printf("       .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,\n");
    printf("       .memory = vkMemory,\n");
    printf("       .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT\n");
    printf("   };\n");
    printf("   vkGetMemoryFdKHR(device, &getFdInfo, &fd);\n\n");

    printf("3. CUDA 导入外部内存:\n");
    printf("   cudaExternalMemoryHandleDesc memDesc = {};\n");
    printf("   memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;\n");
    printf("   memDesc.handle.fd = fd;\n");
    printf("   memDesc.size = size;\n\n");

    printf("   cudaExternalMemory_t extMem;\n");
    printf("   cudaImportExternalMemory(&extMem, &memDesc);\n\n");

    printf("4. 映射到 CUDA 设备指针:\n");
    printf("   cudaExternalMemoryBufferDesc bufDesc = {};\n");
    printf("   bufDesc.offset = 0;\n");
    printf("   bufDesc.size = size;\n\n");

    printf("   void *d_ptr;\n");
    printf("   cudaExternalMemoryGetMappedBuffer(&d_ptr, extMem, &bufDesc);\n\n");

    printf("5. 信号量同步:\n");
    printf("   // 导入 Vulkan 信号量\n");
    printf("   cudaExternalSemaphoreHandleDesc semDesc = {};\n");
    printf("   semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;\n");
    printf("   semDesc.handle.fd = semaphoreFd;\n\n");

    printf("   cudaExternalSemaphore_t extSem;\n");
    printf("   cudaImportExternalSemaphore(&extSem, &semDesc);\n\n");

    printf("   // 等待 Vulkan 完成\n");
    printf("   cudaWaitExternalSemaphoresAsync(&extSem, NULL, 1, stream);\n");
    printf("   \n");
    printf("   // CUDA 操作...\n");
    printf("   \n");
    printf("   // 通知 Vulkan 可以继续\n");
    printf("   cudaSignalExternalSemaphoresAsync(&extSem, NULL, 1, stream);\n\n");
}

// ============================================================================
// 第七部分：性能考虑和最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第七部分：性能最佳实践 ===\n\n");

    printf("1. 减少映射/解除映射次数:\n");
    printf("   // 不好: 每帧多次映射\n");
    printf("   for (int i = 0; i < numBuffers; i++) {\n");
    printf("       cudaGraphicsMapResources(1, &resources[i], 0);\n");
    printf("       // 操作\n");
    printf("       cudaGraphicsUnmapResources(1, &resources[i], 0);\n");
    printf("   }\n\n");

    printf("   // 好: 批量映射\n");
    printf("   cudaGraphicsMapResources(numBuffers, resources, 0);\n");
    printf("   for (int i = 0; i < numBuffers; i++) {\n");
    printf("       // 操作\n");
    printf("   }\n");
    printf("   cudaGraphicsUnmapResources(numBuffers, resources, 0);\n\n");

    printf("2. 使用 CUDA Streams 实现重叠:\n");
    printf("   // 双缓冲技术\n");
    printf("   cudaGraphicsMapResources(1, &buffer[current], stream1);\n");
    printf("   processKernel<<<..., stream1>>>(d_buffer[current]);\n");
    printf("   cudaGraphicsUnmapResources(1, &buffer[current], stream1);\n");
    printf("   \n");
    printf("   // 同时渲染上一帧\n");
    printf("   renderBuffer(buffer[!current]);  // OpenGL\n\n");

    printf("3. 选择合适的注册标志:\n");
    printf("   // 只写: 避免不必要的同步\n");
    printf("   cudaGraphicsRegisterFlagsWriteDiscard\n");
    printf("   \n");
    printf("   // 只读: 允许更好的优化\n");
    printf("   cudaGraphicsRegisterFlagsReadOnly\n\n");

    printf("4. 避免频繁注册/注销:\n");
    printf("   // 不好: 每帧注册\n");
    printf("   while (running) {\n");
    printf("       cudaGraphicsGLRegisterBuffer(&res, buffer, flags);\n");
    printf("       // 使用\n");
    printf("       cudaGraphicsUnregisterResource(res);\n");
    printf("   }\n\n");

    printf("   // 好: 预先注册\n");
    printf("   cudaGraphicsGLRegisterBuffer(&res, buffer, flags);\n");
    printf("   while (running) {\n");
    printf("       // 使用\n");
    printf("   }\n");
    printf("   cudaGraphicsUnregisterResource(res);\n\n");

    printf("5. 同步策略:\n");
    printf("   // 映射前确保 OpenGL 完成\n");
    printf("   glFinish();  // 或使用 Fence\n");
    printf("   cudaGraphicsMapResources(...);\n");
    printf("   \n");
    printf("   // 解除映射后确保 CUDA 完成\n");
    printf("   cudaGraphicsUnmapResources(...);\n");
    printf("   cudaStreamSynchronize(stream);\n");
    printf("   // 然后 OpenGL 可以安全使用\n\n");

    printf("6. 错误处理:\n");
    printf("   cudaError_t err;\n");
    printf("   err = cudaGraphicsMapResources(1, &resource, 0);\n");
    printf("   if (err != cudaSuccess) {\n");
    printf("       // 可能原因:\n");
    printf("       // - OpenGL 仍在使用该资源\n");
    printf("       // - 资源已被映射\n");
    printf("       // - 设备不匹配\n");
    printf("       printf(\"映射失败: %%s\\n\", cudaGetErrorString(err));\n");
    printf("   }\n\n");
}

// ============================================================================
// 第八部分：完整示例框架
// ============================================================================

void demoCompleteExample() {
    printf("=== 第八部分：完整示例框架 ===\n\n");

    printf("// 完整的 CUDA-OpenGL 互操作应用框架\n\n");

    printf("#include <GL/glew.h>\n");
    printf("#include <GL/freeglut.h>\n");
    printf("#include <cuda_runtime.h>\n");
    printf("#include <cuda_gl_interop.h>\n\n");

    printf("// 全局变量\n");
    printf("GLuint pbo;  // Pixel Buffer Object\n");
    printf("GLuint texture;  // 显示纹理\n");
    printf("cudaGraphicsResource *cudaPboResource;\n");
    printf("int width = 1024, height = 768;\n\n");

    printf("// CUDA 渲染内核\n");
    printf("__global__ void renderKernel(uchar4 *output, int w, int h, float t) {\n");
    printf("    int x = threadIdx.x + blockIdx.x * blockDim.x;\n");
    printf("    int y = threadIdx.y + blockIdx.y * blockDim.y;\n");
    printf("    if (x >= w || y >= h) return;\n");
    printf("    \n");
    printf("    int idx = y * w + x;\n");
    printf("    float u = (float)x / w;\n");
    printf("    float v = (float)y / h;\n");
    printf("    \n");
    printf("    // 生成分形图案\n");
    printf("    output[idx] = make_uchar4(\n");
    printf("        (u * 255), (v * 255), (sinf(t) * 127 + 128), 255);\n");
    printf("}\n\n");

    printf("// 初始化\n");
    printf("void initCUDAGL() {\n");
    printf("    // 选择 CUDA 设备\n");
    printf("    cudaGLSetGLDevice(0);\n");
    printf("    \n");
    printf("    // 创建 PBO\n");
    printf("    glGenBuffers(1, &pbo);\n");
    printf("    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);\n");
    printf("    glBufferData(GL_PIXEL_UNPACK_BUFFER,\n");
    printf("                 width * height * 4, NULL, GL_DYNAMIC_DRAW);\n");
    printf("    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);\n");
    printf("    \n");
    printf("    // 注册到 CUDA\n");
    printf("    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo,\n");
    printf("                                  cudaGraphicsRegisterFlagsWriteDiscard);\n");
    printf("    \n");
    printf("    // 创建显示纹理\n");
    printf("    glGenTextures(1, &texture);\n");
    printf("    glBindTexture(GL_TEXTURE_2D, texture);\n");
    printf("    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,\n");
    printf("                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);\n");
    printf("    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);\n");
    printf("    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);\n");
    printf("}\n\n");

    printf("// 每帧渲染\n");
    printf("void display() {\n");
    printf("    static float time = 0.0f;\n");
    printf("    time += 0.01f;\n");
    printf("    \n");
    printf("    // CUDA 渲染到 PBO\n");
    printf("    cudaGraphicsMapResources(1, &cudaPboResource, 0);\n");
    printf("    \n");
    printf("    uchar4 *d_output;\n");
    printf("    size_t size;\n");
    printf("    cudaGraphicsResourceGetMappedPointer(\n");
    printf("        (void**)&d_output, &size, cudaPboResource);\n");
    printf("    \n");
    printf("    dim3 block(16, 16);\n");
    printf("    dim3 grid((width+15)/16, (height+15)/16);\n");
    printf("    renderKernel<<<grid, block>>>(d_output, width, height, time);\n");
    printf("    \n");
    printf("    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);\n");
    printf("    \n");
    printf("    // 更新纹理\n");
    printf("    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);\n");
    printf("    glBindTexture(GL_TEXTURE_2D, texture);\n");
    printf("    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,\n");
    printf("                    width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);\n");
    printf("    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);\n");
    printf("    \n");
    printf("    // 绘制全屏四边形\n");
    printf("    glClear(GL_COLOR_BUFFER_BIT);\n");
    printf("    glEnable(GL_TEXTURE_2D);\n");
    printf("    glBegin(GL_QUADS);\n");
    printf("    glTexCoord2f(0, 0); glVertex2f(-1, -1);\n");
    printf("    glTexCoord2f(1, 0); glVertex2f( 1, -1);\n");
    printf("    glTexCoord2f(1, 1); glVertex2f( 1,  1);\n");
    printf("    glTexCoord2f(0, 1); glVertex2f(-1,  1);\n");
    printf("    glEnd();\n");
    printf("    \n");
    printf("    glutSwapBuffers();\n");
    printf("    glutPostRedisplay();\n");
    printf("}\n\n");

    printf("// 清理\n");
    printf("void cleanup() {\n");
    printf("    cudaGraphicsUnregisterResource(cudaPboResource);\n");
    printf("    glDeleteBuffers(1, &pbo);\n");
    printf("    glDeleteTextures(1, &texture);\n");
    printf("}\n\n");

    printf("// 主函数\n");
    printf("int main(int argc, char **argv) {\n");
    printf("    glutInit(&argc, argv);\n");
    printf("    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);\n");
    printf("    glutInitWindowSize(width, height);\n");
    printf("    glutCreateWindow(\"CUDA-OpenGL Interop\");\n");
    printf("    \n");
    printf("    glewInit();\n");
    printf("    initCUDAGL();\n");
    printf("    \n");
    printf("    glutDisplayFunc(display);\n");
    printf("    atexit(cleanup);\n");
    printf("    \n");
    printf("    glutMainLoop();\n");
    printf("    return 0;\n");
    printf("}\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA 教程 21: CUDA 与图形 API 互操作 (Graphics Interop)        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    // 检查 CUDA 设备
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoInteropConcepts();
    demoOpenGLInteropAPI();
    demoPBOInteropCode();
    demoVBOParticleSystem();
    demoTextureInterop();
    demoVulkanInterop();
    demoBestPractices();
    demoCompleteExample();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                      学习要点总结                               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. OpenGL 互操作核心 API:\n");
    printf("   ┌────────────────────────────────────┬──────────────────────────┐\n");
    printf("   │ 函数                               │ 功能                     │\n");
    printf("   ├────────────────────────────────────┼──────────────────────────┤\n");
    printf("   │ cudaGraphicsGLRegisterBuffer       │ 注册 OpenGL Buffer       │\n");
    printf("   │ cudaGraphicsGLRegisterImage        │ 注册 OpenGL 纹理         │\n");
    printf("   │ cudaGraphicsMapResources           │ 映射资源                 │\n");
    printf("   │ cudaGraphicsResourceGetMappedPointer│ 获取设备指针            │\n");
    printf("   │ cudaGraphicsUnmapResources         │ 解除映射                 │\n");
    printf("   │ cudaGraphicsUnregisterResource     │ 注销资源                 │\n");
    printf("   └────────────────────────────────────┴──────────────────────────┘\n\n");

    printf("2. 常见用途:\n");
    printf("   - PBO: 像素数据处理和显示\n");
    printf("   - VBO: 顶点数据 (粒子系统、网格)\n");
    printf("   - 纹理: 图像处理、动态生成\n\n");

    printf("3. 性能要点:\n");
    printf("   - 批量映射多个资源\n");
    printf("   - 使用合适的注册标志\n");
    printf("   - 避免频繁注册/注销\n");
    printf("   - 使用双缓冲实现重叠\n\n");

    printf("4. Vulkan 互操作特点:\n");
    printf("   - 使用外部内存/信号量\n");
    printf("   - 更底层的控制\n");
    printf("   - 需要显式同步\n\n");

    printf("注意: 完整运行此教程需要 OpenGL 环境\n");
    printf("编译: nvcc 21_interop_graphics.cu -o 21_interop_graphics -lGL -lGLU -lglut\n\n");

    return 0;
}
