#include <ultrahdr_api.h>

#include <inttypes.h>
#include <stdio.h>
#include <memory>

#include <png++/png.hpp>

int main(int argc, char * argv[])
{
	if (argc != 3) {
		fprintf(stderr, "png_to_ultrahdr input.png output.jpg\n");
		return 1;
	}

	const char * inputFilename = argv[1];
	const char * outputFilename = argv[2];

	png::image< png::rgb_pixel > inImg(inputFilename);

	uhdr_codec_private_t* encoder = uhdr_create_encoder();

	uhdr_raw_image_t img;
	img.fmt = UHDR_IMG_FMT_32bppRGBA1010102;
	img.cg = UHDR_CG_BT_2100;
	img.ct = UHDR_CT_PQ;
	img.range = UHDR_CR_FULL_RANGE;

	img.w = inImg.get_width();
	img.h = inImg.get_height();

	auto buffer = std::make_unique<uint8_t[]>( img.w*img.h*4 );

	img.planes[0] = buffer.get();
	img.planes[1] = nullptr;
	img.planes[2] = nullptr;
	img.stride[0] = img.w;
	img.stride[1] = 0;
	img.stride[2] = 0;


	int w = inImg.get_width();
	int h = inImg.get_height();
	for (int iy=0; iy<h; iy++)
		for (int ix=0; ix<w; ix++)
		{
			auto p = buffer.get() + w * iy * 4 + ix * 4;
			
			auto pix = inImg[iy][ix];
			uint16_t r = (pix.red   << 2) + 4;//(pix.red >> 6);
			uint16_t g = (pix.green << 2) + 4;//(pix.green >> 6);
			uint16_t b = (pix.blue  << 2) + 4;//(pix.blue >> 6);
			p[0] = (r << 0) & 0xFF;
			p[1] = (r >> 8) + (g << 2) & 0xFF;
			p[2] = (g >> 6) + (b << 2) & 0xFF;
			p[3] = (b >> 4);
		}
		
	uhdr_enc_set_quality(encoder, 97, UHDR_BASE_IMG);
	uhdr_enc_set_quality(encoder, 97, UHDR_GAIN_MAP_IMG);

	std::cout << "Start set raw\n";
	auto set_error = uhdr_enc_set_raw_image(encoder, &img, UHDR_HDR_IMG);
	if (UHDR_CODEC_OK != set_error.error_code)
	{
		std::cout << "set raw failed\n";
		std::cout << set_error.detail;
		return -1;
	}
	std::cout << "set raw end\n";
	
	std::cout << "Start encode\n";
	auto encode_result = uhdr_encode(encoder);
	if (encode_result.error_code != UHDR_CODEC_OK)
	{
		std::cout << encode_result.detail;
		return -1;
	}
	std::cout << "Encode end\n";
	
	auto compressed = uhdr_get_encoded_stream(encoder);
	if (!compressed)
	{
		std::cout << "no compressed data";
		return -1;
	}
	
	std::cout << "Size: " << compressed->data_sz << std::endl;
	
	FILE * f = fopen(outputFilename, "wb");
	size_t bytesWritten = fwrite(compressed->data, 1, compressed->data_sz, f);
	fclose(f);
	
	
	

/*
        // Fill your RGB(A) data here
        memset(rgb.pixels, 255, rgb.rowBytes * image->height);
		  int w = inImg.get_width();
		  int h = inImg.get_height();
		  for (int iy=0; iy<h; iy++)
			  for (int ix=0; ix<w; ix++)
			  {
				  auto p = rgb.pixels + rgb.rowBytes * iy + ix*4;
				  p[0] = inImg[iy][ix].red;
				  p[1] = inImg[iy][ix].green;
				  p[2] = inImg[iy][ix].blue;
			  }
*/

	uhdr_release_encoder(encoder);
	return 0;
}