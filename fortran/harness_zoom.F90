!
! pydyn zoom harness: run one production zoom routine on a multi-region case
! written by pydyn.zoom_harness (Python) and write every region back.
!
!   harness_zoom<N> <case_dir>        (N = NREGIONS, the case's number of regions)
!
! <case_dir>/case.nml     namelist &zoom: the kernel, the region it is called for,
!                         and the region tree (what dims_grid and initexit's
!                         determine_children_etc / define_splitorderzoom provide in
!                         production; computed by pydyn.zoom and passed in here)
! <case_dir>/<name>_r<k>.bin   unformatted stream, Fortran order, full halo extents:
!     m                (-1:im+2, -1:jm+2, lm)
!     rm rxm rym rzm   (-1:im+2, -1:jm+2, lm, nt)
!     am bm cm         (0:im+1, 0:jm+1, 0:lm+1)
!     pu (0:im, jm, lm)   pv (im, jm+1, lm)   dxyp (jm)
!     sp (-1:im+2, -1:jm+2, 1)   phlb (-1:im+2, -1:jm+2, lm+1)
!     zoomed edge (im, jm) int32
!     entu detu entd detd (im, jm, lm)   dkg (im, jm, lmax_conv)   cloud_* (im, jm) int32
!   and at.bin, bt.bin (lm+1).
! Outputs: out_<name>_r<k>.bin for m rm rxm rym rzm am bm cm sp phlb of every region,
! and out_stats.txt with nloop_max/xi/nxi of every region.
!
! kernel = 'x' | 'y' | 'z'        advectxzoom / advectyzoom / advectzzoom(target), with
!                                 status(:) and splitorderzoom(:,:) from the namelist
!        = 'put_xedges' | 'put_yedges' | 'mix_edges' | 'update_parent'
!        = 'coarsen_region' | 'coarsen_ambm' | 'm2phlb' | 'm2phlb1'
!        = 'dynam0'                dynam0(target) with ndyn
!        = 'convec'                convec(target) with nconv = ndyn
!        = 'massflow'              advectm_cfl.F90's Setup_MassFlow (TimeInterpolation a no-op:
!                                  pu/pv are taken as given)
!        = 'proces_region'         modelIntegration's Proces_Region(target) from the
!                                  statuses given: the recursion over the children, do_steps
!                                  with put_x/yedges at block ends, update_parent + m2phlb
!        = 'step'                  one ndyn step as tm5.F90 runs it: status(1:nreg) = 0,
!                                  then twice itaur(:) = itau ; Proces_Region(1)
!        = 'check_cfl'             advectm_cfl.F90's Init_CFL (cache off) and Check_CFL over
!                                  n_steps steps of ndyn in an interval of nread_; writes
!                                  out_cfl.txt with the ndyn and n it settles on.  pu/pv are
!                                  held over the interval (TimeInterpolation is a no-op here)
! For 'convec', 'proces_region' and 'step' every region with an entu_r<k>.bin gets its
! convection inputs read; the others keep zero fluxes (convec then only applies dkg = 0).
!
#include "tm5.inc"

program harness_zoom

  use dims,         only : dims_init, dims_set_region, nregions, children, ndyn, nconv, limits
  use dims,         only : okdebug, revert, status, splitorderzoom, nloop_max, xi, nxi, at, bt
  use chem_param,   only : ntracet
  use ParTools,     only : ntracetloc, ntracet_ar, lmloc
  use MeteoData,    only : m_dat, pu_dat, pv_dat, sp_dat, phlb_dat
  use MeteoData,    only : entu_dat, detu_dat, entd_dat, detd_dat
  use tracer_data,  only : mass_dat
  use global_data,  only : wind_dat, region_dat, conv_dat
  use advectx,      only : advectxzoom, put_xedges
  use advecty,      only : advectyzoom, put_yedges
  use advectz,      only : advectzzoom
  use advect,       only : dynam0
  use convection,   only : convec
  use zoom_tools,   only : mix_edges, update_parent, coarsen_region
  use advect_tools, only : coarsen_ambm, m2phlb, m2phlb1
  use modelIntegration, only : Proces_Region
  use advectm_cfl,  only : Init_CFL, Check_CFL, Setup_MassFlow
  use dims,         only : nread
  use dims,         only : itau, itaur
  use GO,           only : TDate

  implicit none

  integer, parameter :: mr = NREGIONS

  ! namelist
  character(len=16)  :: kernel = ''
  integer :: target = 1, nreg = 1, nt = 1, lm_ = 0, ndyn_ = 3600, lmax_conv = 0
  integer :: n_steps = 1, nread_ = 3600
  logical :: limits_ = .true., debug = .false.
  integer, dimension(mr) :: im_ = 0, jm_ = 0, parent_ = 0, xcyc_ = 1, touch_sp_ = 1, touch_np_ = 1
  integer, dimension(mr) :: xref_ = 1, yref_ = 1, zref_ = 1, tref_ = 1
  integer, dimension(mr) :: ibeg_ = 1, iend_ = 1, jbeg_ = 1, jend_ = 1, lbeg_ = 1, lend_ = 1
  integer, dimension(mr) :: isr_ = 1, ier_ = 1, jsr_ = 1, jer_ = 1, status_ = 0
  character(len=120), dimension(mr) :: splitorder_ = ''
  integer :: children_(mr, 0:mr) = 0
  namelist /zoom/ kernel, target, nreg, nt, lm_, ndyn_, lmax_conv, limits_, debug, n_steps, nread_, &
                  im_, jm_, parent_, xcyc_, touch_sp_, touch_np_, xref_, yref_, zref_, tref_, &
                  ibeg_, iend_, jbeg_, jend_, lbeg_, lend_, isr_, ier_, jsr_, jer_, &
                  status_, splitorder_, children_

  character(len=1024) :: dir
  integer :: r, n, k, istat, u, imr, jmr, nhalf
  logical :: there
  type(TDate) :: tr(2)

  interface read_bin
    procedure :: read_r1, read_r3, read_r4, read_i2
  end interface
  interface write_bin
    procedure :: write_r3, write_r4
  end interface

  if ( command_argument_count() < 1 ) then
    write (0,*) 'usage: harness_zoom <case_dir>'
    error stop 1
  end if
  call get_command_argument( 1, dir )

  open( newunit=u, file=trim(dir)//'/case.nml', status='old', action='read' )
  read( u, nml=zoom )
  close( u )
  if ( nreg /= nregions ) then   ! production loops over all nregions (Check_CFL, store_masses)
    write (0,*) 'harness_zoom: case has', nreg, 'regions, binary built for', nregions
    error stop 1
  end if
  if ( lmax_conv == 0 ) lmax_conv = lm_

  ! --- module state: the region tree ----------------------------------------
  call dims_init( im_(1), jm_(1), lm_, lmax_conv )
  do r = 1, nreg
    call dims_set_region( r, im_(r), jm_(r), lm_, xref_(r), yref_(r), zref_(r), tref_(r), &
                          parent_(r), xcyc_(r), touch_sp_(r), touch_np_(r), &
                          ibeg_(r), iend_(r), jbeg_(r), jend_(r), lbeg_(r), lend_(r), &
                          isr_(r), ier_(r), jsr_(r), jer_(r) )
    status(r) = status_(r)
    do k = 1, len_trim(splitorder_(r))
      splitorderzoom(r,k) = splitorder_(r)(k:k)
    end do
  end do
  children = 0
  children(1:nreg, 0:nreg) = children_(1:nreg, 0:nreg)
  ndyn = ndyn_ ; nconv = ndyn_
  limits = limits_
  okdebug = debug
  revert = 1
  ntracet = nt ; ntracetloc = nt ; ntracet_ar(0) = nt ; lmloc = lm_
  call read_bin( 'at', at )
  call read_bin( 'bt', bt )

  ! --- every region's arrays -----------------------------------------------
  do r = 1, nreg
    imr = im_(r) ; jmr = jm_(r)
    allocate( m_dat(r)%data(-1:imr+2, -1:jmr+2, lm_) ) ; m_dat(r)%used = .true.
    allocate( mass_dat(r)%rm_t (-1:imr+2, -1:jmr+2, lm_, nt) )
    allocate( mass_dat(r)%rxm_t(-1:imr+2, -1:jmr+2, lm_, nt) )
    allocate( mass_dat(r)%rym_t(-1:imr+2, -1:jmr+2, lm_, nt) )
    allocate( mass_dat(r)%rzm_t(-1:imr+2, -1:jmr+2, lm_, nt) )
    allocate( wind_dat(r)%am_t(0:imr+1, 0:jmr+1, 0:lm_+1) )
    allocate( wind_dat(r)%bm_t(0:imr+1, 0:jmr+1, 0:lm_+1) )
    allocate( wind_dat(r)%cm_t(0:imr+1, 0:jmr+1, 0:lm_+1) )
    allocate( pu_dat(r)%data(0:imr, 1:jmr, 1:lm_) ) ; pu_dat(r)%used = .true.
    allocate( pv_dat(r)%data(1:imr, 1:jmr+1, 1:lm_) ) ; pv_dat(r)%used = .true.
    allocate( sp_dat(r)%data(-1:imr+2, -1:jmr+2, 1) ) ; sp_dat(r)%used = .true.
    allocate( phlb_dat(r)%data(-1:imr+2, -1:jmr+2, lm_+1) ) ; phlb_dat(r)%used = .true.
    allocate( entu_dat(r)%data(imr, jmr, lm_), detu_dat(r)%data(imr, jmr, lm_) )
    allocate( entd_dat(r)%data(imr, jmr, lm_), detd_dat(r)%data(imr, jmr, lm_) )
    entu_dat(r)%used = .true. ; detu_dat(r)%used = .true.
    entd_dat(r)%used = .true. ; detd_dat(r)%used = .true.
    allocate( region_dat(r)%zoomed(imr, jmr), region_dat(r)%edge(imr, jmr), region_dat(r)%dxyp(jmr) )
    allocate( conv_dat(r)%dkg(imr, jmr, lmax_conv), conv_dat(r)%blh(imr, jmr) )
    allocate( conv_dat(r)%cloud_base(imr, jmr), conv_dat(r)%cloud_top(imr, jmr), conv_dat(r)%cloud_lfs(imr, jmr) )
    conv_dat(r)%dkg = 0.0 ; conv_dat(r)%blh = 0.0
    conv_dat(r)%cloud_base = 0 ; conv_dat(r)%cloud_top = 0 ; conv_dat(r)%cloud_lfs = 0
    entu_dat(r)%data = 0.0 ; detu_dat(r)%data = 0.0 ; entd_dat(r)%data = 0.0 ; detd_dat(r)%data = 0.0

    call read_bin( rn('m', r),   m_dat(r)%data )
    call read_bin( rn('rm', r),  mass_dat(r)%rm_t )
    call read_bin( rn('rxm', r), mass_dat(r)%rxm_t )
    call read_bin( rn('rym', r), mass_dat(r)%rym_t )
    call read_bin( rn('rzm', r), mass_dat(r)%rzm_t )
    call read_bin( rn('am', r),  wind_dat(r)%am_t )
    call read_bin( rn('bm', r),  wind_dat(r)%bm_t )
    call read_bin( rn('cm', r),  wind_dat(r)%cm_t )
    call read_bin( rn('pu', r),  pu_dat(r)%data )
    call read_bin( rn('pv', r),  pv_dat(r)%data )
    call read_bin( rn('sp', r),  sp_dat(r)%data )
    call read_bin( rn('phlb', r), phlb_dat(r)%data )
    call read_bin( rn('dxyp', r), region_dat(r)%dxyp )
    call read_bin( rn('zoomed', r), region_dat(r)%zoomed )
    call read_bin( rn('edge', r),   region_dat(r)%edge )
    inquire( file=trim(dir)//'/'//trim(rn('entu', r))//'.bin', exist=there )
    if ( there ) then
      call read_bin( rn('entu', r), entu_dat(r)%data )
      call read_bin( rn('detu', r), detu_dat(r)%data )
      call read_bin( rn('entd', r), entd_dat(r)%data )
      call read_bin( rn('detd', r), detd_dat(r)%data )
      call read_bin( rn('dkg', r),  conv_dat(r)%dkg )
      call read_bin( rn('cloud_top', r),  conv_dat(r)%cloud_top )
      call read_bin( rn('cloud_lfs', r),  conv_dat(r)%cloud_lfs )
      call read_bin( rn('cloud_base', r), conv_dat(r)%cloud_base )
    end if
  end do

  ! --- the call ---------------------------------------------------------------
  select case ( trim(kernel) )
  case ( 'x' )
    call advectxzoom( target )
  case ( 'y' )
    call advectyzoom( target )
  case ( 'z' )
    call advectzzoom( target )
  case ( 'put_xedges' )
    call put_xedges( target )
  case ( 'put_yedges' )
    call put_yedges( target )
  case ( 'mix_edges' )
    call mix_edges( target )
  case ( 'update_parent' )
    call update_parent( target )
  case ( 'coarsen_region' )
    call coarsen_region( target )
  case ( 'coarsen_ambm' )
    call coarsen_ambm( target )
  case ( 'm2phlb' )
    call m2phlb( target )
  case ( 'm2phlb1' )
    call m2phlb1( target )
  case ( 'dynam0' )
    call dynam0( target )
  case ( 'convec' )
    call convec( target, istat )
    if ( istat /= 0 ) then
      write (0,*) 'harness_zoom: convec returned status', istat
      error stop 2
    end if
  case ( 'massflow' )
    call Setup_MassFlow( tr, istat )
    if ( istat /= 0 ) then
      write (0,*) 'harness_zoom: Setup_MassFlow returned status', istat
      error stop 2
    end if
  case ( 'check_cfl' )
    nread = nread_
    call Init_CFL
    call Check_CFL( tr(1), tr(2), n_steps, istat )
    if ( istat /= 0 ) then
      write (0,*) 'harness_zoom: Check_CFL returned status', istat
      error stop 2
    end if
    open( newunit=u, file=trim(dir)//'/out_cfl.txt', status='replace', action='write' )
    write (u, '("ndyn ",i0)') ndyn
    write (u, '("n ",i0)') n_steps
    close( u )
  case ( 'proces_region' )
    itaur(1:nreg) = itau
    call Proces_Region( target, tr, istat )
    if ( istat /= 0 ) then
      write (0,*) 'harness_zoom: Proces_Region returned status', istat
      error stop 2
    end if
  case ( 'step' )
    status(1:nreg) = 0
    do nhalf = 1, 2
      itaur(:) = itau
      call Proces_Region( 1, tr, istat )
      if ( istat /= 0 ) then
        write (0,*) 'harness_zoom: Proces_Region returned status', istat
        error stop 2
      end if
      itau = itau + ndyn/2
    end do
  case default
    write (0,*) 'harness_zoom: unknown kernel "'//trim(kernel)//'"'
    error stop 1
  end select

  do r = 1, nreg
    call write_bin( rn('m', r),   m_dat(r)%data )
    call write_bin( rn('rm', r),  mass_dat(r)%rm_t )
    call write_bin( rn('rxm', r), mass_dat(r)%rxm_t )
    call write_bin( rn('rym', r), mass_dat(r)%rym_t )
    call write_bin( rn('rzm', r), mass_dat(r)%rzm_t )
    call write_bin( rn('am', r),  wind_dat(r)%am_t )
    call write_bin( rn('bm', r),  wind_dat(r)%bm_t )
    call write_bin( rn('cm', r),  wind_dat(r)%cm_t )
    call write_bin( rn('sp', r),  sp_dat(r)%data )
    call write_bin( rn('phlb', r), phlb_dat(r)%data )
  end do

  open( newunit=u, file=trim(dir)//'/out_stats.txt', status='replace', action='write' )
  do r = 1, nreg
    write (u, '("nloop_max",i0,3(1x,i0))') r, (nloop_max(r,n), n=1,3)
    write (u, '("xi",i0,3(1x,es24.17))') r, (xi(r,n), n=1,3)
    write (u, '("nxi",i0,3(1x,i0))') r, (nxi(r,n), n=1,3)
  end do
  close( u )

contains

  function rn( name, r ) result(s)
    character(len=*), intent(in) :: name
    integer, intent(in)          :: r
    character(len=64)            :: s
    write (s, '(a,"_r",i0)') name, r
  end function rn

  ! stream I/O of one array: <dir>/<name>.bin in, <dir>/out_<name>.bin out

  integer function open_bin( name, out ) result(u)
    character(len=*), intent(in) :: name
    logical, intent(in)          :: out
    if ( out ) then
      open( newunit=u, file=trim(dir)//'/out_'//trim(name)//'.bin', status='replace', action='write', &
            access='stream', form='unformatted' )
    else
      open( newunit=u, file=trim(dir)//'/'//trim(name)//'.bin', status='old', action='read', &
            access='stream', form='unformatted' )
    end if
  end function open_bin

  subroutine read_r1( name, a )
    character(len=*), intent(in) :: name
    real, intent(out)            :: a(:)
    integer :: u
    u = open_bin( name, .false. ) ; read (u) a ; close (u)
  end subroutine read_r1

  subroutine read_r3( name, a )
    character(len=*), intent(in) :: name
    real, intent(out)            :: a(:,:,:)
    integer :: u
    u = open_bin( name, .false. ) ; read (u) a ; close (u)
  end subroutine read_r3

  subroutine read_r4( name, a )
    character(len=*), intent(in) :: name
    real, intent(out)            :: a(:,:,:,:)
    integer :: u
    u = open_bin( name, .false. ) ; read (u) a ; close (u)
  end subroutine read_r4

  subroutine read_i2( name, a )
    character(len=*), intent(in) :: name
    integer, intent(out)         :: a(:,:)
    integer :: u
    u = open_bin( name, .false. ) ; read (u) a ; close (u)
  end subroutine read_i2

  subroutine write_r3( name, a )
    character(len=*), intent(in) :: name
    real, intent(in)             :: a(:,:,:)
    integer :: u
    u = open_bin( name, .true. ) ; write (u) a ; close (u)
  end subroutine write_r3

  subroutine write_r4( name, a )
    character(len=*), intent(in) :: name
    real, intent(in)             :: a(:,:,:,:)
    integer :: u
    u = open_bin( name, .true. ) ; write (u) a ; close (u)
  end subroutine write_r4

end program harness_zoom
