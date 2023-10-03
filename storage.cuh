#include <memory>
#include <iostream>
#include <assert.h>
#include <type_traits>
#include <sstream>

template <typename TError>
void checkCudaError(TError result)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorName(result) << "\n";
    }
}

std::string ptr2str(double *ptr)
{
    std::stringstream ss;
    if (ptr == nullptr)
        ss << "nullptr";
    else
        ss << ptr;
    return ss.str();
}

__global__ void kern_dprint(size_t size, double* data)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    for ( ; idx < size; idx += blockDim.x * gridDim.x)
    {
        printf("\t%lu : %f\n", idx, data[idx]);
    }    
}

template <typename TData>
class Storage
{

public:
    Storage() = delete;

    Storage(size_t size)
        : m_size(size)
    {
        m_hptr = nullptr;
        m_dptr = nullptr;
    }

    void allocate_cpu()
    {
        assert(m_hptr == nullptr);
        m_hptr = (TData*)malloc(m_size * sizeof(TData));
    }

    void free_cpu()
    {
        assert(m_hptr != nullptr);
        free(m_hptr);
        m_hptr = nullptr;
    }

    void allocate_gpu()
    {
        assert(m_dptr == nullptr);
        checkCudaError(cudaMalloc((void**)&m_dptr, m_size * sizeof(TData)));
    }

    void free_gpu()
    {
        assert(m_dptr != nullptr);
        checkCudaError(cudaFree((void*) m_dptr));
    }

    void cpy_h2d()
    {
        assert(m_hptr != nullptr && m_dptr != nullptr);
        checkCudaError(cudaMemcpy((void*)m_dptr, (void*)m_hptr, m_size * sizeof(TData), cudaMemcpyHostToDevice));
    }

    void cpy_d2h()
    {
        assert(m_hptr != nullptr && m_dptr != nullptr);
        checkCudaError(cudaMemcpy((void*)m_hptr, (void*)m_dptr, m_size * sizeof(TData), cudaMemcpyDeviceToHost));
    }

    TData *hptr()
    {
        return m_hptr;
    }

    TData *dptr()
    {
        return m_dptr;
    }

    void print_summary()
    {
        std::cout << "Memory summary:\nCPU = " << ptr2str(m_hptr) << "\nGPU = " << ptr2str(m_dptr) << "\n";
    }

    void print_cpu()
    {
        std::cout << "CPU Memory (" << ptr2str(m_hptr) << ") BEGIN:\n";
        for (size_t idx = 0; idx < m_size; ++idx)
        {
            std::cout << "\t" << idx << " : " << m_hptr[idx] << "\n";
        }
        std::cout << "CPU Memory END\n";
    }

    void print_gpu()
    {
        std::cout << "GPU Memory (" << ptr2str(m_dptr) << ") BEGIN:\n";
        if (std::is_same<TData, double>::value == true)
        {
            kern_dprint<<<1, 1>>>(m_size, m_dptr);
        }
        else
        {
            std::cout << "GPU printing not supported for this datatype\n";
        }
        checkCudaError(cudaDeviceSynchronize());
        std::cout << "GPU Memory END\n";
    }

private:

    const size_t m_size;

    TData *m_hptr;
    TData *m_dptr;
};