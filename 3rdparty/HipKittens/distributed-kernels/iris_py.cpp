#include <iris/iris.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Iris Tensor wrapper class
class IrisTensor {
public:
    void* data;
    std::vector<int64_t> shape;
    size_t dtype_size;
    std::string dtype;
    iris::iris* iris_ctx;

    IrisTensor(void* ptr, std::vector<int64_t> s, size_t ds, const std::string& dt, iris::iris* ctx) 
        : data(ptr), shape(s), dtype_size(ds), dtype(dt), iris_ctx(ctx) {}
    
    ~IrisTensor() {
        if (data && iris_ctx) {
            iris_ctx->deallocate(data);
        }
    }
    
    uintptr_t data_ptr() const { return reinterpret_cast<uintptr_t>(data); }
    std::vector<int64_t> get_shape() const { return shape; }
    std::string get_dtype() const { return dtype; }
    
    size_t numel() const {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }
    
    size_t nbytes() const {
        return numel() * dtype_size;
    }
};

// Iris instance wrapper for Python
class IrisInstance {
private:
    std::shared_ptr<iris::iris> iris_ctx_;
    int rank_;
    int world_size_;

public:
    IrisInstance(size_t heap_size_mb = 1024, bool verbose = false) {
        // Initialize MPI
        auto mpi_result = iris::mpi::initialize();
        rank_ = mpi_result.rank;
        world_size_ = mpi_result.world_size;
        
        // Create iris instance
        size_t heap_size_bytes = heap_size_mb * 1024 * 1024;
        iris_ctx_ = std::make_shared<iris::iris>(heap_size_bytes, rank_, world_size_, verbose);
    }
    
    ~IrisInstance() {
        // Finalize will be called when all iris instances are destroyed
    }
    
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    
    // Create empty tensor (returns IrisTensor)
    std::shared_ptr<IrisTensor> empty(std::vector<int64_t> shape, const std::string& dtype = "bfloat16") {
        size_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        size_t dtype_size;
        if (dtype == "bfloat16" || dtype == "bf16") {
            dtype_size = 2;
        } else if (dtype == "float32" || dtype == "float") {
            dtype_size = 4;
        } else if (dtype == "float16" || dtype == "half") {
            dtype_size = 2;
        } else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
        
        void* ptr = iris_ctx_->allocate<char>(total_elements * dtype_size);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate iris memory");
        }
        
        return std::make_shared<IrisTensor>(ptr, shape, dtype_size, dtype, iris_ctx_.get());
    }
    
    iris::iris* get_context() { return iris_ctx_.get(); }
    
    void barrier() {
        iris_ctx_->barrier();
    }
    
    // Get device view for kernel usage
    iris::iris_device_view get_device_view() {
        return iris_ctx_->get_device_view();
    }
};

PYBIND11_MODULE(iris_py, m) {
    m.doc() = "Iris Python bindings for distributed GPU memory management";
    
    // Bind iris_device_view class
    py::class_<iris::iris_device_view>(m, "IrisDeviceView")
        .def("cur_rank", &iris::iris_device_view::cur_rank, "Get current rank")
        .def("world_size", &iris::iris_device_view::world_size, "Get world size");
    
    // Bind IrisInstance class
    py::class_<IrisInstance, std::shared_ptr<IrisInstance>>(m, "Iris")
        .def(py::init<size_t, bool>(), 
             py::arg("heap_size_mb") = 1024, 
             py::arg("verbose") = false,
             "Create iris instance with MPI initialization\n\n"
             "Args:\n"
             "    heap_size_mb: Size of distributed heap in megabytes (default: 1024)\n"
             "    verbose: Enable verbose logging (default: False)")
        .def("rank", &IrisInstance::rank, "Get current MPI rank")
        .def("world_size", &IrisInstance::world_size, "Get total number of MPI ranks")
        .def("empty", &IrisInstance::empty, 
             py::arg("shape"), 
             py::arg("dtype") = "bfloat16",
             "Create empty tensor allocated with iris\n\n"
             "Args:\n"
             "    shape: Tuple or list of tensor dimensions\n"
             "    dtype: Data type ('bfloat16', 'float32', 'float16')\n\n"
             "Returns:\n"
             "    IrisTensor: Iris tensor object")
        .def("barrier", &IrisInstance::barrier, "Synchronize all MPI ranks")
        .def("get_device_view", &IrisInstance::get_device_view, "Get device view for kernel usage");
    
    // Bind IrisTensor class
    py::class_<IrisTensor, std::shared_ptr<IrisTensor>>(m, "Tensor")
        .def("data_ptr", &IrisTensor::data_ptr, "Get raw data pointer as integer")
        .def_property_readonly("shape", &IrisTensor::get_shape, "Get tensor shape")
        .def_property_readonly("dtype", &IrisTensor::get_dtype, "Get tensor dtype")
        .def("numel", &IrisTensor::numel, "Get total number of elements")
        .def("nbytes", &IrisTensor::nbytes, "Get total size in bytes");
    
    // Bind MPI initialization functions
    m.def("mpi_init", []() {
        auto result = iris::mpi::initialize();
        return py::make_tuple(result.rank, result.world_size);
    }, "Initialize MPI and return (rank, world_size)");
    
    m.def("mpi_finalize", &iris::mpi::finalize, "Finalize MPI");
}

