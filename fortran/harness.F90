!
! pydyn kernel harness: run one production dynamic-core kernel on a case
! written by pydyn.harness (Python) and write the result back.
!
!   harness <case_dir>
!
! <case_dir>/case.nml     namelist &case: kernel, im, jm, lm, nt, ndyn, lmax_conv, limits
! <case_dir>/<name>.bin   unformatted stream, float64 (int32 for cloud_*), Fortran order,
!                         full halo extents as allocated in tm5_tracer_data / global_data:
!     m            (-1:im+2, -1:jm+2, lm)
!     rm rxm rym rzm (-1:im+2, -1:jm+2, lm, nt)
!     am bm cm     (0:im+1, 0:jm+1, 0:lm+1)
!     pu           (0:im, jm, lm)          pv (im, jm+1, lm)        bt (lm+1)
!     entu detu entd detd (im, jm, lm)     dkg (im, jm, lmax_conv)
!     cloud_top cloud_lfs cloud_base (im, jm) int32
! Outputs: out_<name>.bin for the arrays the kernel may modify, and out_stats.txt
! with nloop_max(1,:), xi(1,:), nxi(1,:) from dims.
!
! kernel = 'x' | 'y' | 'z'   advectxzoom/advectyzoom/advectzzoom(1) with
!                            status(1)=0 and splitorderzoom(1,1:6)='xyzvsc',
!                            which resolves to dynamu(1,1,jm,1,lm),
!                            dynamv(1,1,im,1,lm), dynamw(1,1,im,1,jm)
!        = 'dynam0'          dynam0(1): am, bm, cm from pu, pv, bt, ndyn
!        = 'convec'          convec(1, status) with nconv = ndyn
!
#include "tm5.inc"

program harness

  use dims,        only : dims_init, im, jm, lm, ndyn, nconv, limits, bt, okdebug
  use dims,        only : status, splitorderzoom, nloop_max, xi, nxi, revert
  use chem_param,  only : ntracet
  use ParTools,    only : ntracetloc, ntracet_ar, lmloc
  use MeteoData,   only : m_dat, pu_dat, pv_dat, entu_dat, detu_dat, entd_dat, detd_dat
  use tracer_data, only : mass_dat
  use global_data, only : wind_dat, region_dat, conv_dat
  use advectx,     only : advectxzoom
  use advecty,     only : advectyzoom
  use advectz,     only : advectzzoom
  use advect,      only : dynam0
  use convection,  only : convec

  implicit none

  ! namelist
  character(len=16) :: kernel = ''
  integer :: im_ = 0, jm_ = 0, lm_ = 0, nt = 1, ndyn_ = 3600, lmax_conv = 0
  logical :: limits_ = .true., debug = .false.
  namelist /case/ kernel, im_, jm_, lm_, nt, ndyn_, lmax_conv, limits_, debug

  character(len=1024) :: dir
  integer :: n, istat, u

  interface read_bin
    procedure :: read_r1, read_r3, read_r4, read_i2
  end interface
  interface write_bin
    procedure :: write_r3, write_r4
  end interface

  if ( command_argument_count() < 1 ) then
    write (0,*) 'usage: harness <case_dir>'
    error stop 1
  end if
  call get_command_argument( 1, dir )

  open( newunit=u, file=trim(dir)//'/case.nml', status='old', action='read' )
  read( u, nml=case )
  close( u )
  if ( lmax_conv == 0 ) lmax_conv = lm_

  ! --- module state -------------------------------------------------------
  call dims_init( im_, jm_, lm_, lmax_conv )
  ndyn = ndyn_ ; nconv = ndyn_
  limits = limits_
  okdebug = debug
  revert = 1
  ntracet = nt ; ntracetloc = nt ; ntracet_ar(0) = nt ; lmloc = lm_
  status(1) = 0
  splitorderzoom(1,1:6) = (/ 'x', 'y', 'z', 'v', 's', 'c' /)

  allocate( m_dat(1)%data(-1:im_+2, -1:jm_+2, lm_) ) ; m_dat(1)%used = .true.
  allocate( mass_dat(1)%rm_t (-1:im_+2, -1:jm_+2, lm_, nt) )
  allocate( mass_dat(1)%rxm_t(-1:im_+2, -1:jm_+2, lm_, nt) )
  allocate( mass_dat(1)%rym_t(-1:im_+2, -1:jm_+2, lm_, nt) )
  allocate( mass_dat(1)%rzm_t(-1:im_+2, -1:jm_+2, lm_, nt) )
  allocate( wind_dat(1)%am_t(0:im_+1, 0:jm_+1, 0:lm_+1) )
  allocate( wind_dat(1)%bm_t(0:im_+1, 0:jm_+1, 0:lm_+1) )
  allocate( wind_dat(1)%cm_t(0:im_+1, 0:jm_+1, 0:lm_+1) )
  allocate( pu_dat(1)%data(0:im_, 1:jm_, 1:lm_) ) ; pu_dat(1)%used = .true.
  allocate( pv_dat(1)%data(1:im_, 1:jm_+1, 1:lm_) ) ; pv_dat(1)%used = .true.
  allocate( entu_dat(1)%data(im_, jm_, lm_), detu_dat(1)%data(im_, jm_, lm_) )
  allocate( entd_dat(1)%data(im_, jm_, lm_), detd_dat(1)%data(im_, jm_, lm_) )
  entu_dat(1)%used = .true. ; detu_dat(1)%used = .true.
  entd_dat(1)%used = .true. ; detd_dat(1)%used = .true.
  allocate( region_dat(1)%zoomed(im_, jm_), region_dat(1)%edge(im_, jm_), region_dat(1)%dxyp(jm_) )
  region_dat(1)%zoomed = 1 ; region_dat(1)%edge = -1 ; region_dat(1)%dxyp = 1.0
  allocate( conv_dat(1)%dkg(im_, jm_, lmax_conv), conv_dat(1)%blh(im_, jm_) )
  allocate( conv_dat(1)%cloud_base(im_, jm_), conv_dat(1)%cloud_top(im_, jm_), conv_dat(1)%cloud_lfs(im_, jm_) )
  conv_dat(1)%dkg = 0.0 ; conv_dat(1)%blh = 0.0
  conv_dat(1)%cloud_base = 0 ; conv_dat(1)%cloud_top = 0 ; conv_dat(1)%cloud_lfs = 0
  wind_dat(1)%am_t = 0.0 ; wind_dat(1)%bm_t = 0.0 ; wind_dat(1)%cm_t = 0.0
  pu_dat(1)%data = 0.0 ; pv_dat(1)%data = 0.0

  ! --- inputs, kernel, outputs ---------------------------------------------
  call read_bin( 'm', m_dat(1)%data )
  call read_bin( 'rm',  mass_dat(1)%rm_t )
  call read_bin( 'rxm', mass_dat(1)%rxm_t )
  call read_bin( 'rym', mass_dat(1)%rym_t )
  call read_bin( 'rzm', mass_dat(1)%rzm_t )

  select case ( trim(kernel) )
  case ( 'x' )
    call read_bin( 'am', wind_dat(1)%am_t )
    call advectxzoom( 1 )
    call write_bin( 'am', wind_dat(1)%am_t )
  case ( 'y' )
    call read_bin( 'bm', wind_dat(1)%bm_t )
    call advectyzoom( 1 )
    call write_bin( 'bm', wind_dat(1)%bm_t )
  case ( 'z' )
    call read_bin( 'cm', wind_dat(1)%cm_t )
    call advectzzoom( 1 )
    call write_bin( 'cm', wind_dat(1)%cm_t )
  case ( 'dynam0' )
    call read_bin( 'pu', pu_dat(1)%data )
    call read_bin( 'pv', pv_dat(1)%data )
    call read_bin( 'bt', bt )
    call dynam0( 1 )
    call write_bin( 'am', wind_dat(1)%am_t )
    call write_bin( 'bm', wind_dat(1)%bm_t )
    call write_bin( 'cm', wind_dat(1)%cm_t )
  case ( 'convec' )
    call read_bin( 'entu', entu_dat(1)%data )
    call read_bin( 'detu', detu_dat(1)%data )
    call read_bin( 'entd', entd_dat(1)%data )
    call read_bin( 'detd', detd_dat(1)%data )
    call read_bin( 'dkg',  conv_dat(1)%dkg )
    call read_bin( 'cloud_top',  conv_dat(1)%cloud_top )
    call read_bin( 'cloud_lfs',  conv_dat(1)%cloud_lfs )
    call read_bin( 'cloud_base', conv_dat(1)%cloud_base )
    call convec( 1, istat )
    if ( istat /= 0 ) then
      write (0,*) 'harness: convec returned status', istat
      error stop 2
    end if
  case default
    write (0,*) 'harness: unknown kernel "'//trim(kernel)//'"'
    error stop 1
  end select

  call write_bin( 'm', m_dat(1)%data )
  call write_bin( 'rm',  mass_dat(1)%rm_t )
  call write_bin( 'rxm', mass_dat(1)%rxm_t )
  call write_bin( 'rym', mass_dat(1)%rym_t )
  call write_bin( 'rzm', mass_dat(1)%rzm_t )

  open( newunit=u, file=trim(dir)//'/out_stats.txt', status='replace', action='write' )
  write (u, '("nloop_max",3(1x,i0))') (nloop_max(1,n), n=1,3)
  write (u, '("xi",3(1x,es24.17))') (xi(1,n), n=1,3)
  write (u, '("nxi",3(1x,i0))') (nxi(1,n), n=1,3)
  close( u )

contains

  ! stream I/O of one array: <dir>/<name>.bin in, <dir>/out_<name>.bin out

  integer function open_bin( name, out ) result(u)
    character(len=*), intent(in) :: name
    logical, intent(in)          :: out
    if ( out ) then
      open( newunit=u, file=trim(dir)//'/out_'//name//'.bin', status='replace', action='write', &
            access='stream', form='unformatted' )
    else
      open( newunit=u, file=trim(dir)//'/'//name//'.bin', status='old', action='read', &
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

end program harness
