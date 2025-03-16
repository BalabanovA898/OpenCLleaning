#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <stdlib.h>

#include "Color.h"

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/opencl.hpp>
#endif

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#ifdef DEV
#define IN "/home/andrey/Codes/paradigms/parallel/images/Исходные/300x300.png"
#define OUT "/home/andrey/Codes/paradigms/parallel/sepia/300x300.png"
#endif

int main(int argc, const char **argv)
{
    try
    {
        std::vector<cl::Platform> platforms{};
        cl::Platform::get(&platforms);

        std::vector<cl::Device> devices{};
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);

#ifndef DEV
        if (argc != 3)
        {
            std::cerr << "Incorrect usage. Correct one is ./sepia <input.png> <output.png>" << std::endl;
            return 1;
        }

        std::string in_image_path = argv[1];
        std::string out_image_path = argv[2];
#endif

#ifdef DEV
        std::string in_image_path = IN;
        std::string out_image_path = OUT;
#endif
        std::clock_t start = std::clock();

        int width, height, channels;
        unsigned char *data = stbi_load(in_image_path.c_str(), &width, &height, &channels, 0);
        if (!data)
        {
            std::cerr << "Ошибка загрузки изображения" << std::endl;
            return 1;
        }
        size_t img_size = width * height * channels;

        std::unique_ptr<Color[]> image_src(new Color[width * height]);
        std::unique_ptr<Color[]> image_res(new Color[width * height]);

        for (size_t i = 0; i < img_size; i += channels)
        {
            image_src[i / channels] = Color{
                *(data + i + 0),
                *(data + i + 1),
                *(data + i + 2),
            };
        }

        cl::Buffer clInputVector = cl::Buffer(context, CL_MEM_READ_ONLY, width * height * sizeof(Color));
        cl::Buffer clOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE, width * height * sizeof(Color));

        queue.enqueueWriteBuffer(clInputVector, CL_FALSE, 0, width * height * sizeof(Color), image_src.get());
        queue.enqueueWriteBuffer(clOutputVector, CL_FALSE, 0, width * height * sizeof(Color), image_res.get());

        std::ifstream sourceFile("./sepia_kernel.cl");
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(cl::vector<cl::string>{sourceCode});

        cl::Program program = cl::Program(context, source);
        program.build(devices);

        cl::Kernel kernel(program, "sepia");

        kernel.setArg(0, clInputVector);
        kernel.setArg(1, clOutputVector);
        kernel.setArg(2, width * height);

        cl::NDRange global(width * height * sizeof(Color));
        cl::NDRange local(4);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();

        std::unique_ptr<Color[]> result(new Color[width * height]);

        queue.enqueueReadBuffer(clOutputVector, CL_TRUE, 0, width * height * sizeof(Color), result.get());

        unsigned char *buffer = (unsigned char *)malloc(img_size);

        for (size_t i = 0; i < width * height; ++i)
        {
            buffer[i * channels + 0] = result[i].r;
            buffer[i * channels + 1] = result[i].g;
            buffer[i * channels + 2] = result[i].b;
        }

        if (!stbi_write_jpg(out_image_path.c_str(), width, height, channels, buffer, 90))
        {
            std::cout << "Ошибка сохранения изображения" << std::endl;
            stbi_image_free(data);
            return 1;
        }

        stbi_image_free(data);

        delete[] result.get();
        free(buffer);

        std::clock_t end = std::clock();
        std::cout << double(end - start) / (double)CLOCKS_PER_SEC << std::endl;
    }
    catch (cl::Error err)
    {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return (EXIT_FAILURE);
    }

    return 0;
}
