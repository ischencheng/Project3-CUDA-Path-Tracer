#include "image.h"

#include <stb_image_write.h>

#include <cstdio>
#include <iostream>
#include <string>

Image::Image(int x, int y)
    : xSize(x), ySize(y), pixels(new glm::vec3[x * y]) 
{}

Image::~Image()
{
    delete[] pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel)
{
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename)
{
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++)
    {
        for (int x = 0; x < xSize; x++)
        {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

void Image::savePFM(const std::string &baseFilename)
{
    std::string filename = baseFilename + ".pfm";
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f)
    {
        std::cout << "Could not write " << filename << std::endl;
        return;
    }
    fprintf(f, "PF\n%d %d\n-1.0\n", xSize, ySize);
    // PFM stores rows bottom to top.
    for (int y = ySize - 1; y >= 0; y--)
    {
        fwrite(&pixels[y * xSize], sizeof(glm::vec3), xSize, f);
    }
    fclose(f);
    std::cout << "Saved " + filename + "." << std::endl;
}
