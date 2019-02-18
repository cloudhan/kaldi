#ifndef _VFS_PROVIDER_H_
#define _VFS_PROVIDER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "util/miniz.h"

#include <istream>
#include <streambuf>

struct membuf : std::streambuf {
  membuf(char const* base, size_t size) {
    char* p(const_cast<char*>(base));
    this->setg(p, p, p + size);
  }
};
struct imemstream : virtual membuf, std::istream {
  imemstream(char const* base, size_t size)
      : membuf(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

namespace VFS {

class VFSProvider {
 public:
  VFSProvider() = delete;
  VFSProvider(const VFSProvider&) = delete;
  VFSProvider(std::vector<uint8_t>&& mem);

  ~VFSProvider();

  bool HasFile(const std::string& name);

  // size_t GetFileSize(std::string name);
  // size_t GetFileSize(uint32_t file_index);

  std::vector<uint8_t> GetFile(const std::string& name);

 private:
  std::vector<uint8_t> memory;
  mz_zip_archive zip_archive;
  std::unordered_map<std::string, size_t> name2index;
  std::unordered_map<std::string, size_t> name2size;
};

VFSProvider* Get();
void Set(VFSProvider* provider);
void Unset();

}  // namespace VFS

#endif
