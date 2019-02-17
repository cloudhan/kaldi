#include "vfs_provider.h"

#include <iostream>

namespace VFS
{

static VFSProvider* provider;

VFSProvider::VFSProvider(std::vector<uint8_t>&& mem)
    : memory(std::move(mem))
// , name2index()
// , name2size()
{
    memset(&zip_archive, 0, sizeof(zip_archive));

    mz_bool status = mz_zip_reader_init_mem(&zip_archive, memory.data(), memory.size(), 0);
    if (!status)
    {
        mz_zip_error err = mz_zip_get_last_error(&zip_archive);
        std::cout << "ERROR:" << mz_zip_get_error_string(err) << std::endl;
        return;
    }

    std::cout << "INFO:\n";
    size_t num_file = mz_zip_reader_get_num_files(&zip_archive);
    name2index.reserve(num_file);
    name2size.reserve(num_file);
    for (size_t i = 0; i < num_file; i++)
    {
        mz_zip_archive_file_stat stat;
        mz_zip_reader_file_stat(&zip_archive, i, &stat);
        std::string filename(stat.m_filename);
        name2index[filename] = i;
        name2size[filename]  = stat.m_uncomp_size;

        std::cout << "filename: " << filename << "\n\tindex: " << i << "\n\tuncompressed size:" << stat.m_uncomp_size
                  << std::endl;
    }
}

VFSProvider::~VFSProvider()
{
    mz_zip_reader_end(&zip_archive);
}

bool VFSProvider::HasFile(const std::string& name)
{
    auto stripped_name = name.substr(2, name.size() - 2);
    return name2index.find(stripped_name) != name2index.end();
}

std::vector<uint8_t> VFSProvider::GetFile(const std::string& name)
{
    auto stripped_name = name.substr(2, name.size() - 2);
    size_t file_index = name2index.at(stripped_name);
    size_t size       = name2size.at(stripped_name);
    std::vector<uint8_t> file(size);
    mz_zip_reader_extract_to_mem(&zip_archive, file_index, file.data(), size, 0);
    std::cout << "successfully read file " << stripped_name << " from VFS" << std::endl;

    return file;
}

VFSProvider* Get()
{
    return ::VFS::provider;
}

void Set(VFSProvider* provider)
{
    ::VFS::provider = provider;
}

void Unset()
{
    ::VFS::provider = nullptr;
}

} // end of namespace VFS
