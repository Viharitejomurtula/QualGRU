#ifndef QUALGRU_FORMAT_HPP

#define QUALGRU_FORMAT_HPP



#include <cstdint>

#include <cstddef>

#include <vector>

#include <stdexcept>

#include <zlib.h>



// All functions are `inline` because they are DEFINED in a header. Without it,

// two .cpp files including this produce duplicate symbols at link time -- which

// works fine until the moment compress and decompress become separate files.



//We are representing read lengths across the 186k reads

//If the read length is under 128 then we use 1 byte

//If the read length is between 128 and 16383 then we use 2 bytes

//If the read length is between 16383 and 2097151 then we use 3 bytes

//This way we aren't wasting bytes which are just 0s for smaller read lengths

//////////////PSUEDOCODE/////////////////////////////////////////

//will have a byte stream to read from

//shift = 0

//value = 0

//loop:

        //byte = next_byte

        //value = value + ((byte & 0x7F) << shift)

        //if (byte & 0x80) == 0: stop loop

        //shift = shift + 7



inline uint32_t read_varint(const uint8_t *&ptr) {

        uint32_t value = 0;

        int shift = 0;

        while (true) {

                uint8_t byte = *ptr;

                ptr++;

                value += (uint32_t)(byte & 0x7F) << shift;

                if ((byte & 0x80) == 0) break;

                shift += 7;

        }

        return value;

}



inline void write_varint(std::vector<uint8_t> &out, uint32_t value) {

        while (true) {

                if (value >= 128) {

                        out.push_back((value & 0x7F) | 0x80);

                        value >>= 7;

                }

                else {

                        out.push_back(value);

                        break;

                }

        }

}



//Need to write a read/write function converting header<->bytes

//for writing from a number to bytes:

        //byte n = (10000 >> n*8) & 0xFF

        //push byte n and increase n by 1

//for reading from bytes to a number:

        //n = byte n << 8*n

        //add values at each time accounting for base to reconstruct whole number



inline void write_u32(std::vector<uint8_t> &out, uint32_t value) {

        for (int i = 0; i < 4; i++) {

                out.push_back((value >> (8 * i)) & 0xFF);

        }

}



inline uint32_t read_u32(const uint8_t *&ptr) {

        uint32_t value = 0;

        for (int i = 0; i < 4; i++) {

                value += (uint32_t)(*ptr++) << (8 * i);

        }

        return value;

}



inline void write_u64(std::vector<uint8_t> &out, uint64_t value) {

        for (int i = 0; i < 8; i++) {

                out.push_back((value >> (8 * i)) & 0xFF);

        }

}



inline uint64_t read_u64(const uint8_t *&ptr) {

        uint64_t value = 0;

        for (int i = 0; i < 8; i++) {

                value += (uint64_t)(*ptr++) << (8 * i);

        }

        return value;

}





struct Header {

    uint8_t  version;

    uint8_t  model_id;              // 0 = h64, 1 = h256, 2 = h32

    uint8_t  flags;                 // bit 0: sequences present

    uint64_t n_reads;

    uint64_t n_symbols;             // total quality characters

    uint32_t crc_quals;

    uint32_t crc_seqs;

    uint64_t seq_block_len;         // COMPRESSED size of the sequence block

    uint64_t seq_uncompressed_len;  // original size -- zlib streams do not

                                    // carry it, so the decoder needs it here

                                    // to know how much to allocate

    uint64_t qual_block_len;

};



// 4 magic + version + model_id + flags

// + n_reads(8) + n_symbols(8) + crc_quals(4) + crc_seqs(4)

// + seq_block_len(8) + seq_uncompressed_len(8) + qual_block_len(8)

static constexpr size_t HEADER_SIZE = 4 + 1 + 1 + 1 + 8 + 8 + 4 + 4 + 8 + 8 + 8;



inline void write_header(std::vector<uint8_t> &out, const Header &h) {

        out.push_back('Q');

        out.push_back('G');

        out.push_back('R');

        out.push_back('U');

        out.push_back(h.version);

        out.push_back(h.model_id);

        out.push_back(h.flags);

        write_u64(out, h.n_reads);

        write_u64(out, h.n_symbols);

        write_u32(out, h.crc_quals);

        write_u32(out, h.crc_seqs);

        write_u64(out, h.seq_block_len);

        write_u64(out, h.seq_uncompressed_len);

        write_u64(out, h.qual_block_len);

}



inline bool read_header(const uint8_t *&ptr, size_t available, Header &h) {

        // size check FIRST -- checking magic on a 2-byte buffer already reads

        // past the end

        if (available < HEADER_SIZE) return false;

        if (ptr[0] != 'Q' || ptr[1] != 'G' || ptr[2] != 'R' || ptr[3] != 'U') return false;

        ptr += 4;



        h.version = *ptr++;

        if (h.version != 1) return false;   // don't silently misread a future version



        h.model_id             = *ptr++;

        h.flags                = *ptr++;

        h.n_reads              = read_u64(ptr);

        h.n_symbols            = read_u64(ptr);

        h.crc_quals            = read_u32(ptr);

        h.crc_seqs             = read_u32(ptr);

        h.seq_block_len        = read_u64(ptr);

        h.seq_uncompressed_len = read_u64(ptr);

        h.qual_block_len       = read_u64(ptr);

        return true;

}





// Compress a byte vector with zlib. Used for the sequence stream and the

// read-name stream -- neither is modelled, and zlib is fine for both.

inline std::vector<uint8_t> zlib_compress(const std::vector<uint8_t> &in) {

        uLongf bound = compressBound(in.size());

        std::vector<uint8_t> out(bound);



        int rc = compress2(out.data(), &bound,

                            in.data(), in.size(),

                            Z_BEST_COMPRESSION);

        if (rc != Z_OK) throw std::runtime_error("zlib compress failed");



        out.resize(bound);      // bound now holds the ACTUAL compressed size

        return out;

}



// Decompress. The caller must know the original size -- zlib streams do not

// carry it, which is why seq_uncompressed_len is stored in the header.

inline std::vector<uint8_t> zlib_decompress(const std::vector<uint8_t> &in,

                                             size_t uncompressed_size) {

        std::vector<uint8_t> out(uncompressed_size);

        uLongf dest_len = uncompressed_size;



        int rc = uncompress(out.data(), &dest_len,

                             in.data(), in.size());

        if (rc != Z_OK) throw std::runtime_error("zlib decompress failed");

        if (dest_len != uncompressed_size) throw std::runtime_error("zlib size mismatch");



        return out;

}



#endif
