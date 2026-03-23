module forway_macro_kernel
    use iso_c_binding
    use omp_lib
    implicit none
    private

    public :: gemm_float_impl, gemm_double_impl

    interface
        subroutine forway_micro_kernel_float(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate) bind(C, name="forway_micro_kernel_float")
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value, intent(in) :: packed_a
            type(c_ptr), value, intent(in) :: packed_b
            type(c_ptr), value :: c_block
            integer(c_size_t), value, intent(in) :: mr, nr, kc, ldc
            integer(c_int), value, intent(in) :: accumulate
        end subroutine

        subroutine forway_micro_kernel_double(packed_a, packed_b, c_block, mr, nr, kc, ldc, accumulate) bind(C, name="forway_micro_kernel_double")
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value, intent(in) :: packed_a
            type(c_ptr), value, intent(in) :: packed_b
            type(c_ptr), value :: c_block
            integer(c_size_t), value, intent(in) :: mr, nr, kc, ldc
            integer(c_int), value, intent(in) :: accumulate
        end subroutine
        
        integer(c_size_t) function forway_get_lanes_float() bind(C, name="forway_get_lanes_float")
            import :: c_size_t
        end function
        
        integer(c_size_t) function forway_get_lanes_double() bind(C, name="forway_get_lanes_double")
            import :: c_size_t
        end function
        
        type(c_ptr) function forway_aligned_malloc(size) bind(C, name="forway_aligned_malloc")
            import :: c_size_t, c_ptr
            integer(c_size_t), value :: size
        end function
        
        subroutine forway_aligned_free(ptr) bind(C, name="forway_aligned_free")
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine
    end interface

contains

    subroutine gemm_float_impl(M, N, K, A_cptr, lda, B_cptr, ldb, C_cptr, ldc) bind(C, name="forway_gemm_float")
        integer(c_int), value, intent(in) :: M, N, K, lda, ldb, ldc
        type(c_ptr), value, intent(in) :: A_cptr, B_cptr
        type(c_ptr), value :: C_cptr

        real(c_float), pointer :: A(:), B(:), C(:)
        real(c_float), pointer :: packed_a(:), packed_b(:), c_micro(:,:)
        type(c_ptr) :: raw_a, raw_b, raw_c
        integer :: mc, nc, kc, mr, nr, vec_len
        integer :: i_mc, j_nc, p_kc
        integer :: mc_cur, nc_cur, kc_cur
        integer :: i_mr, j_nr
        integer :: mr_cur, nr_cur
        integer :: num_a_panels, num_b_panels
        integer :: ii, jj, pp
        integer :: a_idx, b_idx, c_idx
        integer :: i_global, j_global
        integer :: j_panel, j_in, i_panel, i_in
        integer(c_int) :: accumulate

        call c_f_pointer(A_cptr, A, [M * K])
        call c_f_pointer(B_cptr, B, [K * N])
        call c_f_pointer(C_cptr, C, [M * N])

        vec_len = int(forway_get_lanes_float())
        mr = 2 * vec_len
        nr = 8
        mc = 128
        kc = 128
        nc = (N + omp_get_max_threads() - 1) / omp_get_max_threads()
        nc = ((nc + nr - 1) / nr) * nr
        if (nc < nr) nc = nr

        !$OMP PARALLEL PRIVATE(j_nc, nc_cur, p_kc, kc_cur, i_mc, mc_cur, num_a_panels, num_b_panels) &
        !$OMP PRIVATE(ii, jj, pp, a_idx, b_idx, c_idx, i_global, j_global, mr_cur, nr_cur) &
        !$OMP PRIVATE(j_panel, j_in, i_panel, i_in, accumulate) &
        !$OMP PRIVATE(packed_a, packed_b, c_micro, raw_a, raw_b, raw_c)

        raw_a = forway_aligned_malloc(int(mr * kc * ((mc + mr - 1) / mr), c_size_t) * 4_c_size_t)
        raw_b = forway_aligned_malloc(int(kc * nr * ((nc + nr - 1) / nr), c_size_t) * 4_c_size_t)
        raw_c = forway_aligned_malloc(int(mr * nr, c_size_t) * 4_c_size_t)

        call c_f_pointer(raw_a, packed_a, [mr * kc * ((mc + mr - 1) / mr)])
        call c_f_pointer(raw_b, packed_b, [kc * nr * ((nc + nr - 1) / nr)])
        call c_f_pointer(raw_c, c_micro, [mr, nr])

        !$OMP DO
        do j_nc = 0, N - 1, nc
            nc_cur = min(nc, N - j_nc)

            do p_kc = 0, K - 1, kc
                kc_cur = min(kc, K - p_kc)

                num_b_panels = (nc_cur + nr - 1) / nr
                do j_panel = 0, num_b_panels - 1
                    do j_in = 0, nr - 1
                        jj = j_panel * nr + j_in
                        if (jj < nc_cur) then
                            do pp = 0, kc_cur - 1
                                b_idx = (p_kc + pp) * ldb + (j_nc + jj)
                                packed_b(j_panel * kc_cur * nr + pp * nr + j_in + 1) = B(b_idx + 1)
                            end do
                        else
                            do pp = 0, kc_cur - 1
                                packed_b(j_panel * kc_cur * nr + pp * nr + j_in + 1) = 0.0
                            end do
                        end if
                    end do
                end do

                do i_mc = 0, M - 1, mc
                    mc_cur = min(mc, M - i_mc)

                    num_a_panels = (mc_cur + mr - 1) / mr
                    do i_panel = 0, num_a_panels - 1
                        do i_in = 0, mr - 1
                            ii = i_panel * mr + i_in
                            if (ii < mc_cur) then
                                do pp = 0, kc_cur - 1
                                    a_idx = (i_mc + ii) * lda + (p_kc + pp)
                                    packed_a(i_panel * kc_cur * mr + pp * mr + i_in + 1) = A(a_idx + 1)
                                end do
                            else
                                do pp = 0, kc_cur - 1
                                    packed_a(i_panel * kc_cur * mr + pp * mr + i_in + 1) = 0.0
                                end do
                            end if
                        end do
                    end do

                    do j_nr = 0, num_b_panels - 1
                        do i_mr = 0, num_a_panels - 1
                            mr_cur = min(mr, mc_cur - i_mr * mr)
                            nr_cur = min(nr, nc_cur - j_nr * nr)

                            i_global = i_mc + i_mr * mr
                            j_global = j_nc + j_nr * nr
                            
                            if (p_kc == 0) then
                                accumulate = 0_c_int
                            else
                                accumulate = 1_c_int
                            end if

                            if (accumulate /= 0_c_int) then
                                do jj = 1, nr_cur
                                    do ii = 1, mr_cur
                                        c_idx = (i_global + ii - 1) * ldc + (j_global + jj - 1)
                                        c_micro(ii, jj) = C(c_idx + 1)
                                    end do
                                end do
                            end if
                            
                            call forway_micro_kernel_float( &
                                c_loc(packed_a(i_mr * kc_cur * mr + 1)), &
                                c_loc(packed_b(j_nr * kc_cur * nr + 1)), &
                                c_loc(c_micro(1, 1)), &
                                int(mr, c_size_t), &
                                int(nr, c_size_t), &
                                int(kc_cur, c_size_t), &
                                int(mr, c_size_t), &
                                accumulate)
                            
                            do jj = 1, nr_cur
                                do ii = 1, mr_cur
                                    c_idx = (i_global + ii - 1) * ldc + (j_global + jj - 1)
                                    C(c_idx + 1) = c_micro(ii, jj)
                                end do
                            end do

                        end do
                    end do
                end do
            end do
        end do
        !$OMP END DO

        call forway_aligned_free(raw_a)
        call forway_aligned_free(raw_b)
        call forway_aligned_free(raw_c)
        !$OMP END PARALLEL
    end subroutine

    subroutine gemm_double_impl(M, N, K, A_cptr, lda, B_cptr, ldb, C_cptr, ldc) bind(C, name="forway_gemm_double")
        integer(c_int), value, intent(in) :: M, N, K, lda, ldb, ldc
        type(c_ptr), value, intent(in) :: A_cptr, B_cptr
        type(c_ptr), value :: C_cptr

        real(c_double), pointer :: A(:), B(:), C(:)
        real(c_double), pointer :: packed_a(:), packed_b(:), c_micro(:,:)
        type(c_ptr) :: raw_a, raw_b, raw_c
        integer :: mc, nc, kc, mr, nr, vec_len
        integer :: i_mc, j_nc, p_kc
        integer :: mc_cur, nc_cur, kc_cur
        integer :: i_mr, j_nr
        integer :: mr_cur, nr_cur
        integer :: num_a_panels, num_b_panels
        integer :: ii, jj, pp
        integer :: a_idx, b_idx, c_idx
        integer :: i_global, j_global
        integer :: j_panel, j_in, i_panel, i_in
        integer(c_int) :: accumulate

        call c_f_pointer(A_cptr, A, [M * K])
        call c_f_pointer(B_cptr, B, [K * N])
        call c_f_pointer(C_cptr, C, [M * N])

        vec_len = int(forway_get_lanes_double())
        mr = 2 * vec_len
        nr = 8
        mc = 128
        kc = 128
        nc = (N + omp_get_max_threads() - 1) / omp_get_max_threads()
        nc = ((nc + nr - 1) / nr) * nr
        if (nc < nr) nc = nr

        !$OMP PARALLEL PRIVATE(j_nc, nc_cur, p_kc, kc_cur, i_mc, mc_cur, num_a_panels, num_b_panels) &
        !$OMP PRIVATE(ii, jj, pp, a_idx, b_idx, c_idx, i_global, j_global, mr_cur, nr_cur) &
        !$OMP PRIVATE(j_panel, j_in, i_panel, i_in, accumulate) &
        !$OMP PRIVATE(packed_a, packed_b, c_micro, raw_a, raw_b, raw_c)

        raw_a = forway_aligned_malloc(int(mr * kc * ((mc + mr - 1) / mr), c_size_t) * 8_c_size_t)
        raw_b = forway_aligned_malloc(int(kc * nr * ((nc + nr - 1) / nr), c_size_t) * 8_c_size_t)
        raw_c = forway_aligned_malloc(int(mr * nr, c_size_t) * 8_c_size_t)

        call c_f_pointer(raw_a, packed_a, [mr * kc * ((mc + mr - 1) / mr)])
        call c_f_pointer(raw_b, packed_b, [kc * nr * ((nc + nr - 1) / nr)])
        call c_f_pointer(raw_c, c_micro, [mr, nr])

        !$OMP DO
        do j_nc = 0, N - 1, nc
            nc_cur = min(nc, N - j_nc)

            do p_kc = 0, K - 1, kc
                kc_cur = min(kc, K - p_kc)

                num_b_panels = (nc_cur + nr - 1) / nr
                do j_panel = 0, num_b_panels - 1
                    do j_in = 0, nr - 1
                        jj = j_panel * nr + j_in
                        if (jj < nc_cur) then
                            do pp = 0, kc_cur - 1
                                b_idx = (p_kc + pp) * ldb + (j_nc + jj)
                                packed_b(j_panel * kc_cur * nr + pp * nr + j_in + 1) = B(b_idx + 1)
                            end do
                        else
                            do pp = 0, kc_cur - 1
                                packed_b(j_panel * kc_cur * nr + pp * nr + j_in + 1) = 0.0_c_double
                            end do
                        end if
                    end do
                end do

                do i_mc = 0, M - 1, mc
                    mc_cur = min(mc, M - i_mc)

                    num_a_panels = (mc_cur + mr - 1) / mr
                    do i_panel = 0, num_a_panels - 1
                        do i_in = 0, mr - 1
                            ii = i_panel * mr + i_in
                            if (ii < mc_cur) then
                                do pp = 0, kc_cur - 1
                                    a_idx = (i_mc + ii) * lda + (p_kc + pp)
                                    packed_a(i_panel * kc_cur * mr + pp * mr + i_in + 1) = A(a_idx + 1)
                                end do
                            else
                                do pp = 0, kc_cur - 1
                                    packed_a(i_panel * kc_cur * mr + pp * mr + i_in + 1) = 0.0_c_double
                                end do
                            end if
                        end do
                    end do

                    do j_nr = 0, num_b_panels - 1
                        do i_mr = 0, num_a_panels - 1
                            mr_cur = min(mr, mc_cur - i_mr * mr)
                            nr_cur = min(nr, nc_cur - j_nr * nr)

                            i_global = i_mc + i_mr * mr
                            j_global = j_nc + j_nr * nr
                            
                            if (p_kc == 0) then
                                accumulate = 0_c_int
                            else
                                accumulate = 1_c_int
                            end if

                            if (accumulate /= 0_c_int) then
                                do jj = 1, nr_cur
                                    do ii = 1, mr_cur
                                        c_idx = (i_global + ii - 1) * ldc + (j_global + jj - 1)
                                        c_micro(ii, jj) = C(c_idx + 1)
                                    end do
                                end do
                            end if
                            
                            call forway_micro_kernel_double( &
                                c_loc(packed_a(i_mr * kc_cur * mr + 1)), &
                                c_loc(packed_b(j_nr * kc_cur * nr + 1)), &
                                c_loc(c_micro(1, 1)), &
                                int(mr, c_size_t), &
                                int(nr, c_size_t), &
                                int(kc_cur, c_size_t), &
                                int(mr, c_size_t), &
                                accumulate)
                            
                            do jj = 1, nr_cur
                                do ii = 1, mr_cur
                                    c_idx = (i_global + ii - 1) * ldc + (j_global + jj - 1)
                                    C(c_idx + 1) = c_micro(ii, jj)
                                end do
                            end do

                        end do
                    end do
                end do
            end do
        end do
        !$OMP END DO

        call forway_aligned_free(raw_a)
        call forway_aligned_free(raw_b)
        call forway_aligned_free(raw_c)
        !$OMP END PARALLEL
    end subroutine

end module forway_macro_kernel
