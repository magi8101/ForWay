module forway_rng_macro_kernel
    use iso_c_binding
    use omp_lib
    implicit none
    private

    public :: forway_random_uniform_float

    interface
        subroutine forway_chacha8_micro_kernel_float(output_array, num_elements, seed, start_counter) &
            bind(C, name="forway_chacha8_micro_kernel_float")
            import :: c_ptr, c_size_t, c_int64_t
            type(c_ptr), value :: output_array
            integer(c_size_t), value, intent(in) :: num_elements
            integer(c_int64_t), value, intent(in) :: seed
            integer(c_int64_t), value, intent(in) :: start_counter
        end subroutine

        function forway_rng_get_lanes_float() bind(C, name="forway_rng_get_lanes_float")
            import :: c_size_t
            integer(c_size_t) :: forway_rng_get_lanes_float
        end function
    end interface

contains

    subroutine forway_random_uniform_float(N, C_cptr, seed) &
        bind(C, name="forway_random_uniform_float")
        integer(c_int), value, intent(in) :: N
        type(c_ptr), value :: C_cptr
        integer(c_int64_t), value, intent(in) :: seed

        real(c_float), pointer :: C(:)
        integer :: max_threads, optimal_threads, tid
        integer(c_size_t) :: chunk_size, start_idx
        integer(c_int64_t) :: start_counter
        integer(c_size_t) :: elements_to_generate
        integer(c_size_t) :: vector_stride

        call c_f_pointer(C_cptr, C, [N])

        max_threads = omp_get_max_threads()
        
        if (N <= 2500000) then
            optimal_threads = 1
        else if (N <= 20000000) then
            optimal_threads = 2
        else if (N <= 150000000) then
            optimal_threads = max(1, max_threads / 2)
        else
            optimal_threads = max_threads
        end if

        ! Calculate dynamically the actual hardware output stride logically mapped without locking boundaries.
        vector_stride = forway_rng_get_lanes_float() * 16_c_size_t

        chunk_size = (N + optimal_threads - 1) / optimal_threads
        chunk_size = ((chunk_size + vector_stride - 1) / vector_stride) * vector_stride 

        !$OMP PARALLEL NUM_THREADS(optimal_threads) PRIVATE(tid, start_idx, start_counter, elements_to_generate)
        tid = omp_get_thread_num()
        
        start_idx = tid * chunk_size
        
        if (start_idx < N) then
            ! A single Thread State geometrically accounts for exactly 16 floating points dynamically mapped per ChaCha block mathematically!
            start_counter = int(start_idx / 16, c_int64_t)

            elements_to_generate = min(chunk_size, int(N - start_idx, c_size_t))
            
            call forway_chacha8_micro_kernel_float( &
                c_loc(C(start_idx + 1)), &
                elements_to_generate, &
                seed, &
                start_counter)
        end if
        !$OMP END PARALLEL
    end subroutine

end module forway_rng_macro_kernel
