pub fn find_optimal_batch_size(_device: &crate::cuda_helpers::CudaDevice) -> Result<usize, String> {
    Ok(1000000)
}
