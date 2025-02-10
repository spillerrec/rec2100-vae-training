#include <ultrahdr_api.h>

#include <inttypes.h>
#include <stdio.h>
#include <memory>

#include <png++/png.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <png.h>
#include <cstdint>
#include <cmath>
#include <optional>

#include <exiv2/exiv2.hpp>

#include "half_float/umHalf.h"

struct PngCleanup {
	png_structp png = nullptr;
	png_infop info = nullptr;
	
	~PngCleanup() {
		if (info)
			png_destroy_info_struct(png, &info);
		if (png)
			png_destroy_read_struct(&png, nullptr, nullptr);
	}
};

struct FileDeleter {
	void operator()(FILE* fp) const { if (fp) fclose(fp); }
};

std::optional<std::string> ReadParametersFromPNG(const char* filename) {
	auto Error = [](auto msg){
		std::cerr << "PNG parameters error: " << msg << std::endl;
		return std::optional<std::string>();
	};
	std::unique_ptr<FILE, FileDeleter> fp(fopen(filename, "rb"));
	if (!fp)
		return Error("Unable to open file");

	unsigned char header[8];
	fread(header, 1, 8, fp.get());
	if (png_sig_cmp(header, 0, 8))
		return Error("Not a PNG file.");

	PngCleanup png;
	png.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png.png)
		return Error("Failed to create PNG read struct.");

	png.info = png_create_info_struct(png.png);
	if (!png.info)
		return Error("Failed to create PNG info struct.");

	if (setjmp(png_jmpbuf(png.png)))
		return Error("Failed to setjmp for PNG error handling.");

	png_init_io(png.png, fp.get());
	png_set_sig_bytes(png.png, 8);
	png_read_info(png.png, png.info);

	std::cout << "Reading PNG chunks from: " << filename << std::endl;

	png_textp text_ptr;
	int num_text;
	if (png_get_text(png.png, png.info, &text_ptr, &num_text) > 0) {
		for (int i = 0; i < num_text; ++i)
			  if (text_ptr[i].key == std::string("parameters"))
				return std::string(reinterpret_cast<char*>(text_ptr[i].text), text_ptr[i].text_length);
	}
	
	return {};
}



std::vector<uint8_t> createExifWithComment(const std::string& comment) {
	Exiv2::Blob exifBuffer;
	Exiv2::ExifData exifData;
	 
	Exiv2::Value::AutoPtr v = Exiv2::Value::create(Exiv2::asciiString);
	v->read(comment);
	Exiv2::ExifKey key("Exif.Photo.UserComment");
	exifData.add(key, v.get());

	Exiv2::ExifParser parser;
	parser.encode(exifBuffer, Exiv2::littleEndian, exifData);
	 
	 
	//std::unique_ptr<FILE, FileDeleter> fp(fopen("meta.exif", "wb"));
	 //fwrite(exifBuffer.data(), exifBuffer.size(), 1, fp.get());
	 
	 std::vector<uint8_t> header = {0x45, 0x78, 0x69, 0x66, 0x00, 0x00};
	 exifBuffer.insert(exifBuffer.begin(), header.begin(), header.end());

	return std::vector<uint8_t>(exifBuffer.data(), exifBuffer.data() + exifBuffer.size());
}

double pq_to_linear(double pq)
{
	double c1 = 0.8359;
	double c2 = 18.8516;
	double c3 = 18.6875;
	double m = 0.1593;
	double n = 78.8438;

	pq = std::clamp(pq, 0.0, 1.0);
	double L = (std::pow(std::max(pq, 0.0), (1.0 / n)) - c1) / (c2 - c3 * (std::pow(std::max(pq, 0.0), (1.0 / n))));
	return std::pow(std::clamp(L, 0.0, 1.0), (1.0 / m)) / 203 * 10000;
}

int main(int argc, char * argv[])
{
	if (argc != 3) {
		fprintf(stderr, "png_to_ultrahdr input.png output.jpg\n");
		return 1;
	}

	const char * inputFilename = argv[1];
	const char * outputFilename = argv[2];
	
	auto parameters = ReadParametersFromPNG(inputFilename);

	png::image< png::rgb_pixel > inImg(inputFilename);

	uhdr_codec_private_t* encoder = uhdr_create_encoder();

	uhdr_raw_image_t img;
	img.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat;
	img.cg = UHDR_CG_BT_2100;
	img.ct = UHDR_CT_LINEAR;
	img.range = UHDR_CR_FULL_RANGE;

	img.w = inImg.get_width();
	img.h = inImg.get_height();

	auto buffer = std::make_unique<half[]>( img.w*img.h*4 );

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
			p[0] = pq_to_linear( pix.red / 255.0 );
			p[1] = pq_to_linear( pix.green / 255.0 );
			p[2] = pq_to_linear( pix.blue / 255.0 );
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
	
	uhdr_mem_block block;
	std::vector<uint8_t> exif;
	if (parameters){
		std::cout << "Writing parameters" << std::endl;
		std::cout << *parameters << std::endl;
		exif = createExifWithComment(*parameters);
		block.data = exif.data();
		
		block.data_sz = block.capacity = exif.size();
		auto exif_result = uhdr_enc_set_exif_data(encoder, &block);
		if (exif_result.error_code != UHDR_CODEC_OK)
			std::cout << exif_result.detail;
	}
	
	uhdr_enc_set_using_multi_channel_gainmap(encoder, 0);
	
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

	uhdr_release_encoder(encoder);
	return 0;
}