!
! Stub modules for the pydyn kernel harness.
!
! The production kernel sources (advectx/y/z__slopes.F90, advect.F90,
! convection.F90) are compiled unmodified against these modules.  Every
! symbol below appears in a `use ..., only :` list of those files for the
! serial, no-budget, no-zoom preprocessor branch; the declarations keep the
! production kinds, ranks and argument lists so that the kernels see the
! same interfaces.  Grid sizes are runtime variables set by `dims_init`
! instead of the compile-time parameters of dims_grid/dims_levels; the
! kernels only use them in executable statements and automatic-array
! declarations, which both accept variables.
!
! Behaviour: nregions = NREGIONS (default 1), no reduced grid, one MPI task,
! one OpenMP thread, okdebug off.  With NREGIONS = 1 (the `harness` binary)
! mix_edges / compress_xedges are exact no-ops for region 1 in the production
! code (zoom_tools.F90 returns before touching anything), so an empty stub is
! faithful.  With NREGIONS > 1 (the `harness_zoom` binary, docs/state.md §3d)
! the real zoom_tools.F90 and advect_tools__speedup.F90 are compiled instead
! and the region tree -- refinement factors, parents, children, child
! footprints, interior scopes -- is set at run time by the driver
! (`dims_set_region`), where production has dims_grid parameters and
! initexit's determine_children_etc.  escape_tm and goExit terminate with a
! nonzero exit code after writing the message to stderr.
!

#include "tm5.inc"

#ifndef NREGIONS
#define NREGIONS 1
#endif

! ---------------------------------------------------------------------------
module GO
  implicit none
  public
  character(len=1024) :: gol = ''
  integer, parameter  :: gol_unit = 6
  character(len=1), parameter :: pathsep = '/'

  ! modelIntegration (harness_zoom only) passes dates to operators the transport-only
  ! build compiles out, and reads one rc key in Proces_Init, which is never called
  type TDate
    integer :: time6(6) = 0
  end type TDate
  type TrcFile
    character(len=256) :: fname = ''
  end type TrcFile
  interface ReadRc
    module procedure ReadRc_s, ReadRc_l
  end interface
  ! Check_CFL (harness_zoom) steps its interval with t1 + IncrDate(sec=...) and hands the
  ! dates to TimeInterpolation (a no-op here) and the CFL cache (off); datetime.F90 keeps a
  ! time window it never uses in the harness
  type TIncrDate
    integer :: sec = 0
  end type TIncrDate
  type T_Time_Window
    integer :: dummy = 0
  end type T_Time_Window
  interface operator(+)
    module procedure date_plus_incr
  end interface
contains
  function IncrDate( sec ) result(d)
    integer, intent(in), optional :: sec
    type(TIncrDate)               :: d
    if ( present(sec) ) d%sec = sec
  end function IncrDate
  function date_plus_incr( t, d ) result(u)
    type(TDate), intent(in)     :: t
    type(TIncrDate), intent(in) :: d
    type(TDate)                 :: u
    u = t ; u%time6(6) = u%time6(6) + d%sec
  end function date_plus_incr
  subroutine Get( t, time6 )
    type(TDate), intent(in)        :: t
    integer, intent(out), optional :: time6(6)
    if ( present(time6) ) time6 = t%time6
  end subroutine Get
  subroutine ReadRc_l( rcF, key, value, status, default )
    type(TrcFile), intent(in)     :: rcF
    character(len=*), intent(in)  :: key
    logical, intent(out)          :: value
    integer, intent(out)          :: status
    logical, intent(in), optional :: default
    value = .false. ; if ( present(default) ) value = default
    status = 0
    if ( .false. ) print *, rcF%fname, key
  end subroutine ReadRc_l
  function NewDate( time6 ) result(t)
    integer, intent(in) :: time6(6)
    type(TDate)         :: t
    t%time6 = time6
  end function NewDate
  subroutine goLabel( label )
    character(len=*), intent(in), optional :: label
    if ( .false. .and. present(label) ) print *, label
  end subroutine goLabel
  subroutine Init( rcF, fname, status )
    type(TrcFile), intent(out)   :: rcF
    character(len=*), intent(in) :: fname
    integer, intent(out)         :: status
    rcF%fname = fname ; status = 0
  end subroutine Init
  subroutine Done( rcF, status )
    type(TrcFile), intent(inout) :: rcF
    integer, intent(out)         :: status
    rcF%fname = '' ; status = 0
  end subroutine Done
  subroutine ReadRc_s( rcF, key, value, status, default )
    type(TrcFile), intent(in)              :: rcF
    character(len=*), intent(in)           :: key
    character(len=*), intent(out)          :: value
    integer, intent(out)                   :: status
    character(len=*), intent(in), optional :: default
    value = '' ; if ( present(default) ) value = default
    status = 0
    if ( .false. ) print *, rcF%fname, key
  end subroutine ReadRc_s
  subroutine goPr
    write (gol_unit, '(a)') trim(gol)
    gol = ''
  end subroutine goPr
  subroutine goErr
    write (0, '("ERROR - ",a)') trim(gol)
    gol = ''
  end subroutine goErr
  subroutine GO_Timer_Init( status )
    integer, intent(out) :: status
    status = 0
  end subroutine GO_Timer_Init
  subroutine GO_Timer_Done( status )
    integer, intent(out) :: status
    status = 0
  end subroutine GO_Timer_Done
  subroutine GO_Timer_Def( itim, name, status )
    integer, intent(out)         :: itim
    character(len=*), intent(in) :: name
    integer, intent(out)         :: status
    itim = 0 ; status = 0
    if ( .false. ) print *, name
  end subroutine GO_Timer_Def
  subroutine GO_Timer_Start( itim, status )
    integer, intent(in)  :: itim
    integer, intent(out) :: status
    status = 0
    if ( .false. ) print *, itim
  end subroutine GO_Timer_Start
  subroutine GO_Timer_End( itim, status )
    integer, intent(in)  :: itim
    integer, intent(out) :: status
    status = 0
    if ( .false. ) print *, itim
  end subroutine GO_Timer_End
end module GO

! ---------------------------------------------------------------------------
module go_system
  implicit none
contains
  subroutine goExit( code )
    integer, intent(in) :: code
    write (0, '("harness: goExit(",i0,")")') code
    if ( code /= 0 ) error stop 3
    stop
  end subroutine goExit
end module go_system

! ---------------------------------------------------------------------------
module dims
  implicit none
  public

  integer, parameter :: nregions     = NREGIONS
  integer, parameter :: nregions_max = NREGIONS
  integer, parameter :: nregions_all = nregions_max + 1
  integer, parameter :: maxref       = 10

  ! grid sizes and the region tree (dims_grid parameters in production;
  ! variables here so that one binary serves every configuration):
  integer :: im(0:nregions_all) = 0
  integer :: jm(0:nregions_all) = 0
  integer :: lm(0:nregions_all) = 0
  integer, dimension(0:nregions_max) :: xref = 1, yref = 1, zref = 1, tref = 1
  integer :: parent(nregions_max) = 0
  integer :: xcyc(nregions)     = 1
  integer :: touch_sp(nregions) = 1
  integer :: touch_np(nregions) = 1
  integer :: children(nregions, 0:nregions) = 0
  logical :: newsrun = .true.
  integer :: ibeg(nregions) = 1, iend(nregions) = 1, jbeg(nregions) = 1, jend(nregions) = 1
  integer :: lbeg(nregions) = 1, lend(nregions) = 1
  integer :: isr(nregions) = 1, ier(nregions) = 1, jsr(nregions) = 1, jer(nregions) = 1
  real    :: dx = 6.0, dy = 4.0
  integer :: zbeg(nregions_max) = 0
  integer :: zend(nregions_max) = 0
  real, parameter :: epsz = 0.0001

  ! levels:
  integer            :: lmax_conv = 0
  real, allocatable  :: bt(:), at(:)

  ! operator splitting:
  integer, parameter :: nsplitsteps = 12
  integer, parameter :: n_operators = nsplitsteps/2
  character, dimension(nregions, maxref*nsplitsteps) :: splitorderzoom = ' '
  integer, dimension(nregions) :: status = 0
  logical, parameter :: zoom2D = .true.

  ! constants and diagnostics:
  real, parameter :: one  = 1.0
  real, parameter :: zero = 0.0
  logical :: limits = .true.
  integer, dimension(nregions, 3) :: nxi = 0
  integer, dimension(nregions, 3) :: nloop_max = 0
  real,    dimension(nregions, 3) :: xi = 0.0
  logical :: okdebug = .false.
  integer, parameter :: kdebug = 9
  integer :: revert = 1
  integer :: itau = 0
  integer :: itaur(nregions) = 0
  integer, parameter :: iglbsfc = nregions + 1
  integer, parameter :: zoom_mode = 1

  ! time bookkeeping of dims.F90 that Check_CFL and the real datetime.F90 (harness_zoom) name;
  ! defaults as in base/trunk/src/dims.F90
  integer :: nread = 6*3600, ndyn_max = 3600*3, nsrce = 24*3600, nchem = 0
  integer, dimension(6) :: idate = 0, idatei = 0
  integer :: nstep = 0, kmain = 6
  logical :: newyr = .false., newmonth = .false., newday = .false., cdebug = .false.
  logical :: newhour(nregions) = .false.
  integer :: julday0 = 0, iyear0 = 2000, icalendo = 2
  integer :: ndyn = 3600, nconv = 3600

contains

  subroutine dims_init( imr, jmr, lmr, lmaxc )
    integer, intent(in) :: imr, jmr, lmr, lmaxc
    im = imr ; jm = jmr ; lm = lmr
    im(0) = imr ; jm(0) = jmr ; lm(0) = lmr   ! "parent" of region 1 (parent(1)=0)
    lmax_conv = lmaxc
    zend = lmr
    iend = imr ; jend = jmr ; lend = lmr
    ier  = imr ; jer  = jmr
    allocate( bt(lmr+1), at(lmr+1) ) ; bt = 0.0 ; at = 0.0
  end subroutine dims_init

  ! One region of a zoom configuration: what dims_grid and determine_children_etc
  ! provide in production.  Region 1 must be set first (it fills region 0 too).
  subroutine dims_set_region( r, imr, jmr, lmr, xr, yr, zr, tr, par, cyc, sp, np, &
                              ib, ie, jb, je, lb, le, is_, ie_, js_, je_ )
    integer, intent(in) :: r, imr, jmr, lmr, xr, yr, zr, tr, par, cyc, sp, np
    integer, intent(in) :: ib, ie, jb, je, lb, le, is_, ie_, js_, je_
    im(r) = imr ; jm(r) = jmr ; lm(r) = lmr
    xref(r) = xr ; yref(r) = yr ; zref(r) = zr ; tref(r) = tr
    parent(r) = par ; xcyc(r) = cyc ; touch_sp(r) = sp ; touch_np(r) = np
    ibeg(r) = ib ; iend(r) = ie ; jbeg(r) = jb ; jend(r) = je ; lbeg(r) = lb ; lend(r) = le
    isr(r) = is_ ; ier(r) = ie_ ; jsr(r) = js_ ; jer(r) = je_
    zbeg(r) = 0 ; zend(r) = lmr
    if ( r == 1 ) then
      im(0) = imr ; jm(0) = jmr ; lm(0) = lmr
    end if
  end subroutine dims_set_region

end module dims

! ---------------------------------------------------------------------------
module chem_param
  implicit none
  public
  integer, parameter :: maxtrace = 64
  integer            :: ntracet = 1
  real, dimension(maxtrace) :: ra = 1.0
end module chem_param

! ---------------------------------------------------------------------------
module MeteoData
  implicit none
  public

  type TMeteoData
    real, pointer :: data(:,:,:) => null()
    logical       :: used = .false.
    logical       :: filled1 = .false., filled2 = .false.
  end type TMeteoData

  ! the grid name Init_CFL folds into the cache fingerprint
  type TLliName
    character(len=32) :: name = ''
  end type TLliName

  integer, parameter, private :: nra = NREGIONS + 1   ! nregions_all
  type(TMeteoData), save, target :: m_dat(nra), sp_dat(nra), sp1_dat(nra), sp2_dat(nra), phlb_dat(nra)
  type(TMeteoData), save, target :: pu_dat(nra), pv_dat(nra), pw_dat(nra)
  type(TMeteoData), save, target :: mfu_dat(nra), mfv_dat(nra), mfw_dat(nra), omega_dat(nra)
  type(TMeteoData), save, target :: entu_dat(nra), entd_dat(nra), detu_dat(nra), detd_dat(nra)
  type(TMeteoData), save, target :: entu_conv_dat(nra), entd_conv_dat(nra)
  type(TMeteoData), save, target :: detu_conv_dat(nra), detd_conv_dat(nra)
  type(TMeteoData), save, target :: cp_dat(nra)
  ! named in modelIntegration's Proces_Init only (never called by the harness)
  type(TMeteoData), save, target :: tsp_dat(nra), spm_dat(nra), temper_dat(nra), humid_dat(nra)
  type(TMeteoData), save, target :: cc_dat(nra), cco_dat(nra), ccu_dat(nra), gph_dat(nra)
  type(TMeteoData), save, target :: oro_dat(nra), pclim_dat(nra)
  ! Setup_MassFlow's pivot fields (never active: b_pivot_active is forced .false.)
  type(TMeteoData), save, target :: am_dat(nra), bm_dat(nra), cm_dat(nra)
  type(TLliName), save :: lli(0:nra)

  interface Set
    module procedure mdat_Set
  end interface

contains

  ! Setup_MassFlow interpolates pu/pv to the step; the harness holds them constant over the
  ! interval, so its cases are ones where that is exact (docs/state.md §3d)
  subroutine TimeInterpolation( md, tr, status )
    use GO, only : TDate
    type(TMeteoData), intent(inout) :: md
    type(TDate), intent(in)         :: tr(2)
    integer, intent(out)            :: status
    status = 0
    if ( .false. ) print *, md%used, tr(1)%time6
  end subroutine TimeInterpolation

  subroutine Alloc_Fields( status )
    integer, intent(out) :: status
    status = 0
  end subroutine Alloc_Fields

  subroutine mdat_Set( md, status, used )
    type(TMeteoData), intent(inout) :: md
    integer, intent(out)            :: status
    logical, intent(in), optional   :: used
    if ( present(used) ) md%used = used
    status = 0
  end subroutine mdat_Set

end module MeteoData

! ---------------------------------------------------------------------------
module tracer_data
  use dims, only : nregions
  implicit none
  public

  type mass_data
    ! tracer-parallel arrays (the ones the serial kernels use) ...
    real, pointer :: rm_t(:,:,:,:)  => null()
    real, pointer :: rxm_t(:,:,:,:) => null()
    real, pointer :: rym_t(:,:,:,:) => null()
    real, pointer :: rzm_t(:,:,:,:) => null()
    ! ... and the level-parallel ones referenced in put_xedges/put_yedges
    ! (never associated here: which_par = 'tracer').
    real, pointer :: rm_k(:,:,:,:)  => null()
    real, pointer :: rxm_k(:,:,:,:) => null()
    real, pointer :: rym_k(:,:,:,:) => null()
    real, pointer :: rzm_k(:,:,:,:) => null()
  end type mass_data

  type(mass_data), save, target :: mass_dat(nregions)

contains

  subroutine Par_Check_Domain( region1, others, to_check )
    integer, intent(in)          :: region1
    character(len=*), intent(in) :: others
    character(len=*), intent(in) :: to_check
    if ( .false. ) print *, region1, others, to_check
  end subroutine Par_Check_Domain

  ! okdebug-only diagnostics of do_steps
  subroutine Par_Check_mass( region, label )
    integer, intent(in)          :: region
    character(len=*), intent(in) :: label
    if ( .false. ) print *, region, label
  end subroutine Par_Check_mass
  subroutine tracer_print( label, region, status )
    character(len=*), intent(in) :: label
    integer, intent(in)          :: region
    integer, intent(out)         :: status
    status = 0
    if ( .false. ) print *, label, region
  end subroutine tracer_print

end module tracer_data

! ---------------------------------------------------------------------------
module global_data
  use dims,        only : nregions
  use tracer_data, only : mass_dat, mass_data
  implicit none
  public

  type wind_data
    real, pointer :: am_t(:,:,:) => null()
    real, pointer :: bm_t(:,:,:) => null()
    real, pointer :: cm_t(:,:,:) => null()
  end type wind_data

  type region_data
    integer, pointer :: zoomed(:,:) => null()
    integer, pointer :: edge(:,:)   => null()
    real,    pointer :: dxyp(:)     => null()
  end type region_data

  type conv_data
    real,    pointer :: dkg(:,:,:)      => null()
    real,    pointer :: blh(:,:)        => null()
    integer, pointer :: cloud_base(:,:) => null()
    integer, pointer :: cloud_top(:,:)  => null()
    integer, pointer :: cloud_lfs(:,:)  => null()
  end type conv_data

  type emis_data
    real, pointer :: surf(:,:) => null()
  end type emis_data

  type(wind_data),   save, target :: wind_dat(nregions)
  type(region_data), save, target :: region_dat(nregions)
  type(conv_data),   save, target :: conv_dat(nregions)
  character(len=256) :: rcfile = ''

contains

  subroutine Alias_Wind_Dat( status )
    integer, intent(out) :: status
    status = 0
  end subroutine Alias_Wind_Dat

end module global_data

! ---------------------------------------------------------------------------
module ParTools
  use dims, only : nregions
  implicit none
  public
  integer :: npes = 1, myid = 0, root = 0, root_k = 0, root_t = 0
  integer :: lmloc = 0, offsetl = 0
  integer :: ntracetloc = 0, offsetn = 0
  integer :: ntracet_ar(0:0) = 0
  character(len=6) :: which_par = 'tracer'
  character(len=6) :: previous_par(nregions) = 'tracer'
  logical :: allocate_mass = .true.
end module ParTools

! ---------------------------------------------------------------------------
module omp_ParTools
  implicit none
  public
  logical :: omp_master = .true.
contains
  subroutine my_omp_domain( std_start, std_end, dom_start, dom_end )
    integer, intent(in)  :: std_start, std_end
    integer, intent(out) :: dom_start, dom_end
    dom_start = std_start ; dom_end = std_end
  end subroutine my_omp_domain
  subroutine my_omp_sum( lvec, gvec )
    real, intent(in)    :: lvec
    real, intent(inout) :: gvec
    gvec = lvec
  end subroutine my_omp_sum
  subroutine my_omp_max( lvec, gvec )
    real, intent(in)    :: lvec
    real, intent(inout) :: gvec
    gvec = lvec
  end subroutine my_omp_max
end module omp_ParTools

! ---------------------------------------------------------------------------
module redgridZoom
  use dims, only : nregions
  implicit none
  public
  integer, parameter :: nredmax = 180
  logical :: grid_reduced = .false.
  integer, dimension(nregions) :: nred = 0
  integer, dimension(nredmax, nregions) :: clustsize = 1, imredj = 0, jred = 0
  integer, dimension(180, nregions)     :: imred = 0
contains
  subroutine uni2red( region )
    integer, intent(in) :: region
    write (0,*) 'harness: uni2red called for region', region
    error stop 4
  end subroutine uni2red
  subroutine uni2red_mf( region )
    integer, intent(in) :: region
    write (0,*) 'harness: uni2red_mf called for region', region
    error stop 4
  end subroutine uni2red_mf
  subroutine red2uni( region, m_uni )
    integer, intent(in) :: region
    real, dimension(:,:,:), intent(in) :: m_uni
    write (0,*) 'harness: red2uni called for region', region, shape(m_uni)
    error stop 4
  end subroutine red2uni
end module redgridZoom

#if NREGIONS == 1
! ---------------------------------------------------------------------------
module zoom_tools
  implicit none
contains
  subroutine mix_edges( region )
    integer, intent(in) :: region
    if ( .false. ) print *, region   ! region 1: production returns without touching data
  end subroutine mix_edges
end module zoom_tools
#endif

! ---------------------------------------------------------------------------
module binas
  implicit none
  real, parameter :: grav = 9.80665    ! base/trunk/src/binas.F90
  real, parameter :: p_global = 1.0e5  ! only named in Proces_Init
  real, parameter :: gamma = 0.0       ! only named (renamed away) in dynamwm
end module binas

! ---------------------------------------------------------------------------
module misctools
  implicit none
contains
  ! only reached under okdebug in m2phlb1; the harness never sets okdebug for it
  subroutine collapse_loop( a1, b1, a2, b2, a3, b3, counter )
    integer, intent(in)  :: a1, b1, a2, b2, a3, b3
    integer, intent(out) :: counter(:,:)
    counter = 0
    write (0,*) 'harness: collapse_loop called', a1, b1, a2, b2, a3, b3
    error stop 4
  end subroutine collapse_loop
end module misctools

! ---------------------------------------------------------------------------
module toolbox
  implicit none
contains
  subroutine escape_tm( msg )
    character(len=*), intent(in) :: msg
    write (0, '("harness: escape_tm: ",a)') trim(msg)
    error stop 2
  end subroutine escape_tm
  elemental integer function lvlpress( region, press, press0 )
    integer, intent(in) :: region
    real, intent(in)    :: press, press0
    lvlpress = 1 + 0*region + int(0.0*press*press0)
  end function lvlpress
end module toolbox

#if NREGIONS == 1
! ---------------------------------------------------------------------------
module datetime
  implicit none
contains
  ! Proces_Region turns itaur into the dates it hands do_steps, which only the operators
  ! the transport-only build compiles out (chemistry, sources) read
  subroutine tau2date( itaux, time6 )
    integer, intent(in)  :: itaux
    integer, intent(out) :: time6(6)
    time6 = 0 ; time6(6) = itaux
  end subroutine tau2date
  subroutine tstamp( kunit, itaux, msg )
    integer, intent(in)          :: kunit, itaux
    character(len=*), intent(in) :: msg
    write (kunit, '(i0,1x,a)') itaux, trim(msg)
  end subroutine tstamp
end module datetime
#endif

#if NREGIONS > 1
! ---------------------------------------------------------------------------
! What modelIntegration.F90 names beyond the dynamic core, for harness_zoom's
! Proces_Region.  All no-ops: user output, the pydyn_dump hooks (production's are
! no-ops too unless pydyn.dump : T), and the sources/sinks module, whose operator the
! transport-only build (without_emission) compiles out of do_steps.
module user_output
  implicit none
contains
  subroutine User_Output_Init( status )
    integer, intent(out) :: status
    status = 0
  end subroutine User_Output_Init
  subroutine user_output_step( region, status )
    integer, intent(in)  :: region
    integer, intent(out) :: status
    status = 0
    if ( .false. ) print *, region
  end subroutine user_output_step
end module user_output

module pydyn_dump
  implicit none
contains
  subroutine pydyn_dump_state( tag, region, status )
    character(len=*), intent(in) :: tag
    integer, intent(in)          :: region
    integer, intent(out)         :: status
    status = 0
    if ( .false. ) print *, tag, region
  end subroutine pydyn_dump_state
  subroutine pydyn_dump_m2phlb( region, status )
    integer, intent(in)  :: region
    integer, intent(out) :: status
    status = 0
    if ( .false. ) print *, region
  end subroutine pydyn_dump_m2phlb
end module pydyn_dump

module sources_sinks
  use GO, only : TDate
  implicit none
contains
  subroutine Sources_Sinks_Init( status )
    integer, intent(out) :: status
    status = 0
  end subroutine Sources_Sinks_Init
  subroutine Sources_Sinks_Done( status )
    integer, intent(out) :: status
    status = 0
  end subroutine Sources_Sinks_Done
  subroutine trace_after_read( status )
    integer, intent(out) :: status
    status = 0
  end subroutine trace_after_read
  subroutine sources_sinks_apply( region, tr, status )
    integer, intent(in)     :: region
    type(TDate), intent(in) :: tr(2)
    integer, intent(out)    :: status
    status = 0
    if ( .false. ) print *, region, tr(1)%time6
  end subroutine sources_sinks_apply
end module sources_sinks
#endif

#if NREGIONS > 1
! ---------------------------------------------------------------------------
! The CFL cache's netCDF wrapper.  advectm_cfl.F90 loads and saves the cache through it;
! the harness never turns the cache on, so none of these is reached.
module MDF
  implicit none
  integer, parameter :: MDF_NETCDF4 = 4, MDF_READ = 0, MDF_REPLACE = 1, MDF_GLOBAL = -1, MDF_INT = 4
contains
  subroutine unreachable( what )
    character(len=*), intent(in) :: what
    write (0,*) 'harness: MDF called: ', what
    error stop 4
  end subroutine unreachable
  subroutine MDF_Open( fname, fmt, mode, ncid, status )
    character(len=*), intent(in) :: fname
    integer, intent(in)  :: fmt, mode
    integer, intent(out) :: ncid, status
    ncid = fmt + mode ; status = 1 ; call unreachable( fname )
  end subroutine MDF_Open
  subroutine MDF_Create( fname, fmt, mode, ncid, status )
    character(len=*), intent(in) :: fname
    integer, intent(in)  :: fmt, mode
    integer, intent(out) :: ncid, status
    ncid = fmt + mode ; status = 1 ; call unreachable( fname )
  end subroutine MDF_Create
  subroutine MDF_Close( ncid, status )
    integer, intent(in)  :: ncid
    integer, intent(out) :: status
    status = ncid ; call unreachable( 'close' )
  end subroutine MDF_Close
  subroutine MDF_EndDef( ncid, status )
    integer, intent(in)  :: ncid
    integer, intent(out) :: status
    status = ncid ; call unreachable( 'enddef' )
  end subroutine MDF_EndDef
  subroutine MDF_Inq_Dim_Length( ncid, name, n, status )
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer, intent(out) :: n, status
    n = ncid ; status = 1 ; call unreachable( name )
  end subroutine MDF_Inq_Dim_Length
  subroutine MDF_Inq_VarID( ncid, name, varid, status )
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid, status
    varid = ncid ; status = 1 ; call unreachable( name )
  end subroutine MDF_Inq_VarID
  subroutine MDF_Def_Dim( ncid, name, n, dimid, status )
    integer, intent(in) :: ncid, n
    character(len=*), intent(in) :: name
    integer, intent(out) :: dimid, status
    dimid = ncid + n ; status = 1 ; call unreachable( name )
  end subroutine MDF_Def_Dim
  subroutine MDF_Def_Var( ncid, name, xtype, dims, varid, status )
    integer, intent(in) :: ncid, xtype, dims(:)
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid, status
    varid = ncid + xtype + size(dims) ; status = 1 ; call unreachable( name )
  end subroutine MDF_Def_Var
  subroutine MDF_Get_Att( ncid, varid, name, value, status )
    integer, intent(in) :: ncid, varid
    character(len=*), intent(in) :: name
    integer, intent(out) :: value, status
    value = ncid + varid ; status = 1 ; call unreachable( name )
  end subroutine MDF_Get_Att
  subroutine MDF_Put_Att( ncid, varid, name, value, status )
    integer, intent(in) :: ncid, varid
    character(len=*), intent(in) :: name
    class(*), intent(in) :: value
    integer, intent(out) :: status
    status = ncid + varid ; call unreachable( name )
    select type ( value )
    end select
  end subroutine MDF_Put_Att
  subroutine MDF_Get_Var( ncid, varid, values, status, start, count )
    integer, intent(in) :: ncid, varid
    integer, intent(inout) :: values(..)
    integer, intent(out) :: status
    integer, intent(in), optional :: start(:), count(:)
    status = ncid + varid + rank(values) ; call unreachable( 'get_var' )
    if ( present(start) .and. present(count) ) status = status + size(start) + size(count)
  end subroutine MDF_Get_Var
  subroutine MDF_Put_Var( ncid, varid, values, status )
    integer, intent(in) :: ncid, varid
    integer, intent(in) :: values(..)
    integer, intent(out) :: status
    status = ncid + varid + rank(values) ; call unreachable( 'put_var' )
  end subroutine MDF_Put_Var
end module MDF

! ---------------------------------------------------------------------------
! The shared CFL cache (tm5-zoom 79b30df3b, cfl.cache.shared : T).  advectm_cfl.F90 uses it
! from that commit on; the harness never turns a cache on, so cfls_enabled stays false.
module cfl_cache_shared
  use MDF, only : unreachable
  implicit none
  logical, save :: cfls_enabled = .false.
contains
  subroutine CFLS_Setup( primary_dir, ndyn_max, nread, gridhash, nloop_x, nloop_y, nloop_z, &
                         require_nloop, flat, gridname, status )
    character(len=*), intent(in) :: primary_dir, gridname
    integer, intent(in)  :: ndyn_max, nread, gridhash, nloop_x, nloop_y, nloop_z
    logical, intent(in)  :: require_nloop, flat
    integer, intent(out) :: status
    status = ndyn_max + nread + gridhash + nloop_x + nloop_y + nloop_z
    if ( require_nloop .or. flat ) status = 1
    call unreachable( primary_dir // gridname )
  end subroutine CFLS_Setup
  subroutine CFLS_Load( year, times, ndyns, n, nmax, status )
    integer, intent(in)    :: year, nmax
    integer, intent(inout) :: times(:,:), ndyns(:)
    integer, intent(out)   :: n, status
    n = 0 ; status = year + nmax + size(times) + size(ndyns) ; call unreachable( 'CFLS_Load' )
  end subroutine CFLS_Load
  subroutine CFLS_Save( dir, times, ndyns, n, status )
    character(len=*), intent(in) :: dir
    integer, intent(in)  :: times(:,:), ndyns(:), n
    integer, intent(out) :: status
    status = n + size(times) + size(ndyns) ; call unreachable( dir )
  end subroutine CFLS_Save
end module cfl_cache_shared
#endif
