import mph
import jpype
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    L_PML: str = "0.3[m]"
    R_PML: str = "1[m]"
    c_host: str = "343.2[m/s]"
    rho_host: str = "1.2044[kg/m^3]"
    p0: str = "1[Pa]"
    R_int: str = "0.5[m]"
    mesh_h: str = "0.01[m]"
    d_up_p: str = "1.9[mm]"
    r_p: str = "5.8[mm]/2"
    R_p: str = "23.9[mm]/2"
    H_p: str = "15.1[mm]"
    d_down_p: str = "2.1[mm]"
    d_wall_p: str = "2.1[mm]"
    freq_start: str = "1900 [Hz]"
    freq_step: str = "5 [Hz]"
    freq_stop: str = "2000 [Hz]"
    z0: str = "0.7[m]"
    R_sp: str = "0.01[m]"


def _prepare_model(model, parameters: ModelParameters):
    """Prepeare physics, geometry, variables, study in the model"""
    # set model parameters
    params = model.param()

    params.set("L_PML", parameters.L_PML)
    params.set("R_PML", parameters.R_PML)
    params.set("c_host", parameters.c_host)
    params.set("rho_host", parameters.rho_host)
    params.set("p0", parameters.p0)
    params.set("R_int", parameters.R_int)
    params.set("mesh_h", parameters.mesh_h)
    params.set("d_up_p", parameters.d_up_p)
    params.set("r_p", parameters.r_p)
    params.set("R_p", parameters.R_p)
    params.set("H_p", parameters.H_p)
    params.set("d_down_p", parameters.d_down_p)
    params.set("d_wall_p", parameters.d_wall_p)
    params.set("freq_start", parameters.freq_start)
    params.set("freq_step", parameters.freq_step)
    params.set("freq_stop", parameters.freq_stop)
    params.set("z0", parameters.z0)
    params.set("R_sp", parameters.R_sp)

    # create a component with geometry
    comp1 = model.component().create("comp1", True)
    geom1 = comp1.geom().create("geom1", jpype.types.JInt(2))
    geom1.axisymmetric(True)

    # create circle for host
    c1 = geom1.create("c1", "Circle")
    c1.label("Host")
    c1.set("layername", "Layer 1")
    c1.setIndex("layer", "L_PML", jpype.types.JInt(0))
    c1.set("r", "R_PML+L_PML")

    # create circle for particle
    c2 = geom1.create("c2", "Circle")
    c2.label("Particle physics")
    c2.set("r", "1.3*sqrt(R_p^2 + H_p^2/4)")
    c2.set("pos", ["0", "-H_p/2"])

    # create integrate boundary
    c3 = geom1.create("c3", "Circle")
    c3.label("Int")
    c3.set("r", "R_int")

    # create polygon for particle shape
    pol1 = geom1.create("pol1", "Polygon")
    pol1.label("Scatterer")
    pol1.set("source", "table")
    pol1.set(
        "table",
        [
            ["0", "0"],
            ["R_p", "0"],
            ["R_p", "-H_p"],
            ["r_p", "-H_p"],
            ["r_p", "-H_p + d_up_p"],
            ["R_p - d_wall_p", "-H_p + d_up_p"],
            ["R_p - d_wall_p", "-d_down_p"],
            ["0", "-d_down_p"],
        ],
    )

    # create spherical speaker
    c4 = geom1.create("c4", "Circle")
    c4.label("Speaker")
    c4.set("pos", ["0", "- z0"])
    c4.set("r", "R_sp")

    # create point for probe
    pt1 = geom1.create("pt1", "Point")
    pt1.label("Probe")
    pt1.set("p", ["0", "R_PML-0.01"])

    # build the geometry
    geom1.run()
    geom1.run("fin")

    # Selection: Integration Surface
    sel_int_surf = comp1.selection().create("sel_int_surf", "Disk")
    sel_int_surf.set("entitydim", jpype.types.JInt(1))
    sel_int_surf.label("Integration Surface")
    sel_int_surf.set("r", "1.01 * R_int")
    sel_int_surf.set("rin", "0.99 * R_int")
    sel_int_surf.set("condition", "allvertices")

    # Selection: Scatterer
    sel_scatterer = comp1.selection().create("sel_scatterer", "Disk")
    sel_scatterer.label("Scatterer Domain")
    sel_scatterer.set("posy", "-H_p/2")
    sel_scatterer.set("r", "1.5*sqrt(R_p^2 + H_p^2/4)")
    sel_scatterer.set("condition", "inside")

    # Selection: PML Domain
    sel_PML = comp1.selection().create("sel_PML", "Disk")
    sel_PML.label("PML Domains")
    sel_PML.set("r", "R_PML + L_PML")
    sel_PML.set("rin", "R_PML")
    sel_PML.set("condition", "allvertices")

    # Selection: Speaker area

    speaker_domain = comp1.selection().create("speaker_domain", "Disk")
    speaker_domain.label("Speaker Domain")
    speaker_domain.set("r", "1.01*R_sp")
    speaker_domain.set("posy", "- z0")
    speaker_domain.set("condition", "inside")

    speaker_boundary = comp1.selection().create("speaker_boundary", "Disk")
    speaker_boundary.set("entitydim", jpype.types.JInt(1))
    speaker_boundary.label("Speaker Boundary")
    speaker_boundary.set("r", "1.01 * R_sp")
    speaker_boundary.set("rin", "0.99 * R_sp")
    speaker_boundary.set("posy", "-z0")
    speaker_boundary.set("condition", "allvertices")

    # Selection: Sample — Boundary + Domain
    sample_boundary = comp1.selection().create("sample_boundary", "Box")
    sample_boundary.label("Sample Boundary")
    sample_boundary.set("entitydim", jpype.types.JInt(1))
    sample_boundary.set("xmin", "0.01*H_p")
    sample_boundary.set("xmax", "R_p*1.01")
    sample_boundary.set("ymin", "-H_p*1.01")
    sample_boundary.set("ymax", "H_p*0.01")
    sample_boundary.set("condition", "intersects")

    sample_domain = comp1.selection().duplicate("sample_domain", sample_boundary.tag())
    sample_domain.label("Sample Domain")
    sample_domain.set("entitydim", jpype.types.JInt(2))
    sample_domain.set("xmin", "-0.01*H_p")
    sample_domain.set("condition", "inside")

    # Composite Selections: Host media
    sel_host_with_PML = comp1.selection().create("sel_host_with_PML", "Complement")
    sel_host_with_PML.label("Host with PML")
    sel_host_with_PML.set("input", [sel_scatterer.tag(), speaker_domain.tag()])

    sel_host = comp1.selection().create("sel_host", "Complement")
    sel_host.label("Host")
    sel_host.set("input", [sample_domain.tag(), speaker_domain.tag()])

    sel_host_sample = comp1.selection().create("sel_host_sample", "Complement")
    sel_host_sample.label("Host near Sample")
    sel_host_sample.set(
        "input", [sample_domain.tag(), sel_host_with_PML.tag(), speaker_domain.tag()]
    )

    # Selection: point probe
    sel_probe = comp1.selection().create("sel_probe", "Disk")
    sel_probe.set("entitydim", jpype.types.JInt(0))
    sel_probe.label("Probe")
    sel_probe.set("r", "0.005")
    sel_probe.set("posy", "R_PML-0.01")
    sel_probe.set("condition", "allvertices")

    # Selection: line
    sel_line = comp1.selection().create("sel_line", "Box")
    sel_line.set("entitydim", jpype.types.JInt(1))
    sel_line.label("line")
    sel_line.set("xmin", "-0.01*R_p")
    sel_line.set("xmax", "0.01*R_p")
    sel_line.set("ymin", "-0.01*d_down_p")
    sel_line.set("ymax", "R_PML+ 0.01*d_down_p")
    sel_line.set("condition", "inside")

    # create PML
    pml1 = comp1.coordSystem().create("pml1", "PML")
    pml1.selection().named(sel_PML.tag())

    # create host material
    mat_host = comp1.material().create("mat_host", "Common")
    mat_host.selection().named(sel_host.tag())

    prop = mat_host.propertyGroup("def")

    # ---- Functions for air material properties ----

    # dynamic viscosity η(T)
    eta = prop.func().create("eta", "Piecewise")
    eta.set("arg", "T")
    eta.set(
        "pieces",
        [
            [
                "200",
                "1600",
                "-8.38278e-7 + 8.35717342e-8*T - 7.69429583e-11*T^2 + "
                "4.6437266e-14*T^3 - 1.06585607e-17*T^4",
            ]
        ],
    )
    eta.set("argunit", "K")
    eta.set("fununit", "Pa*s")

    # heat capacity Cp(T)
    Cp = prop.func().create("Cp", "Piecewise")
    Cp.set("arg", "T")
    Cp.set(
        "pieces",
        [
            [
                "200",
                "1600",
                "1047.63657 - 0.372589265*T + 9.45304214e-4*T^2 - "
                "6.02409443e-7*T^3 + 1.2858961e-10*T^4",
            ]
        ],
    )
    Cp.set("argunit", "K")
    Cp.set("fununit", "J/(kg*K)")

    # thermal conductivity k(T)
    k = prop.func().create("k", "Piecewise")
    k.set("arg", "T")
    k.set(
        "pieces",
        [
            [
                "200",
                "1600",
                "-0.00227583562 + 1.15480022e-4*T - 7.90252856e-8*T^2 + "
                "4.11702505e-11*T^3 - 7.43864331e-15*T^4",
            ]
        ],
    )
    k.set("argunit", "K")
    k.set("fununit", "W/(m*K)")

    # bulk viscosity μB(T) = 0.6 η(T)
    muB = prop.func().create("muB", "Analytic")
    muB.set("expr", "0.6*eta(T)")
    muB.set("args", "T")
    muB.set("argunit", "K")
    muB.set("fununit", "Pa*s")

    # ---- Apply functions for air material properties ----
    prop.set("density", "rho_host")
    prop.set("soundspeed", "c_host")
    prop.set("heatcapacity", "Cp(T)")
    prop.set("ratioofspecificheat", "1.4")
    prop.set("dynamicviscosity", "eta(T)")
    prop.set("bulkviscosity", "muB(T)")
    prop.set(
        "thermalconductivity", ["k(T)", "0", "0", "0", "k(T)", "0", "0", "0", "k(T)"]
    )

    prop.addInput("temperature")
    prop.addInput("pressure")

    # create integration operation
    intop1 = comp1.cpl().create("intop1", "Integration")
    intop1.selection().named(sel_int_surf.tag())
    intop1.set("axisym", True)

    # create variables
    variables1 = comp1.variable().create("var1")
    variables1.set("I0", "p0^2/(2 * c_host * rho_host)", "Plane wave intensity")
    variables1.set("dps_dn", "(d(acpr.p_s,z)*z+d(acpr.p_s,r)*r) / R_int", "dp_s / dn")
    variables1.set("dpb_dn", "(d(acpr.p_b,z)*z+d(acpr.p_b,r)*r) / R_int", "dp_s / dn")
    variables1.set(
        "W_ext",
        "1 / (2*acpr.omega*rho_host) * intop1(imag(conj(acpr.p_s) * dpb_dn + conj(acpr.p_b)*dps_dn))",
        "Extinction cross section",
    )
    variables1.set(
        "W_sc",
        "-1 / (2*acpr.omega*rho_host) * intop1(imag(conj(acpr.p_s) * dps_dn))",
        "Scattering cross section",
    )

    # Create variables for spherical wave
    variables2 = comp1.variable().create("var2")
    variables2.set("omega", "2*pi*freq")
    variables2.set("k", "omega/c_host")
    variables2.set("R0", "sqrt(r^2 + (z + z0)^2)")
    variables2.set("p_b", "p0*z0/R0*exp(-i*k*R0)")
    variables2.set("common", "- i*k - 1/R0")
    variables2.set("dp_dr", "p_b*common*(r/R0)")
    variables2.set("dp_dz", "p_b*common*(z + z0)/R0")
    variables2.set("u_br", "dp_dr/(i*omega*rho_host)")
    variables2.set("u_bz", "dp_dz/(i*omega*rho_host)")

    # ACPR
    acpr = comp1.physics().create("acpr", "PressureAcoustics", geom1.tag())
    acpr.selection().named(sel_host_with_PML.tag())
    acpr.prop("cref").set("cref", "c_host")

    pml1.set("wavelengthSource", "acpr")

    # create background pressure field
    bpf1 = acpr.create("bpf1", "BackgroundPressureField", jpype.types.JInt(2))
    bpf1.selection().named(sel_host_with_PML.tag())
    bpf1.set("p", "p_b")
    bpf1.set("CalculateIntensity", True)
    bpf1.set("v", [["u_br"], ["0"], ["u_bz"]])
    bpf1.set("PressureFieldType", "UserDefined")
    bpf1.set("PressureAmplitudeSpherical", "p0 * z0 /(1 [m])")
    bpf1.set("z0", "- z0")
    bpf1.set("c", "c_host")

    # Spherical wave radiation
    swr1 = acpr.create("swr1", "SphericalWaveRadiation", jpype.types.JInt(1))
    swr1.selection().named(speaker_boundary.tag())
    swr1.set("z0", "- z0")

    # TA
    ta = comp1.physics().create("ta", "ThermoacousticsSinglePhysics", geom1.tag())
    ta.selection().named(sel_host_sample.tag())
    ta.prop("cref").set("cref", "c_host")

    # create background acoustic fields
    baf1 = ta.create("baf1", "BackgroundAcousticFields", jpype.types.JInt(2))
    baf1.selection().named(sel_host_sample.tag())
    baf1.set("AcousticFieldType", "UserDefined")
    baf1.set("p", "p_b")
    baf1.set("u", [["u_br"], ["0"], ["u_bz"]])

    # create multiphysics
    atb = comp1.multiphysics().create(
        "atb1", "AcousticThermoacousticBoundary", jpype.types.JInt(1)
    )
    comp1.multiphysics("atb1").selection().all()

    # create mesh
    mesh1 = comp1.mesh().create("mesh1")
    size = mesh1.feature("size")
    size.set("custom", "on")
    size.set("hmax", "mesh_h")
    size.set("hmin", "0.01*mesh_h")
    size1 = mesh1.create("size1", "Size")
    size1.selection().named(sel_host_sample.tag())
    size1.set("custom", "on")
    size1.set("hmax", "0.02*mesh_h")
    bl1 = mesh1.create("bl1", "BndLayer")
    bl1.selection().named(sel_host_sample.tag())
    blp1 = bl1.create("blp1", "BndLayerProp")
    blp1.selection().named(sample_boundary.tag())
    ftri1 = mesh1.create("ftri1", "FreeTri")
    ftri1.selection().named(sel_host_with_PML.tag())
    size1.set("hminactive", True)
    size1.set("hmaxactive", True)

    mesh1.run()

    # create probes
    var_sc = comp1.probe().create("var_sc", "GlobalVariable")
    var_ext = comp1.probe().create("var_ext", "GlobalVariable")
    point_ot = comp1.probe().create("point_ot", "Point")
    point_ot.selection().named(sel_probe.tag())

    var_sc.label("Scattering cross-section")
    var_sc.set("probename", "sigma_sc_norm")
    var_sc.set("expr", "W_sc/I0/(pi*R_p^2)")
    var_sc.set("unit", "1")
    var_sc.set("descr", "Scattering cross-section")
    var_sc.set("window", "window1")

    var_ext.label("Extinction cross-section")
    var_ext.set("probename", "sigma_ext_norm")
    var_ext.set("expr", "W_ext/I0/(pi*R_p^2)")
    var_ext.set("unit", "1")
    var_ext.set("descr", "Extinction cross-section")
    var_ext.set("window", "window1")

    point_ot.label("Optical theorem extinction")
    point_ot.set(
        "expr",
        "- 4*pi / acpr.k * imag(acpr.p_s / acpr.p_b) * (z*z0)/(z+z0) /(pi*R_p^2)",
    )
    point_ot.set("unit", "1")
    point_ot.set("descr", "Optical theorem")
    point_ot.set("window", "window1")

    # create stydy
    std1 = model.study().create("std1")
    std1_freq = std1.create("freq", "Frequency")
    std1_freq.set(
        "plist",
        f"range({parameters.freq_start},{parameters.freq_step},{parameters.freq_stop})",
    )
    std1_freq.set("auto_ngen", jpype.types.JInt(15))
    std1_freq.set("manual_ngen", jpype.types.JInt(15))
    std1_freq.set("auto_ngenactive", False)
    std1_freq.set("manual_ngenactive", False)

    tbl2 = model.result().table().create("tbl2", "Table")
    tbl2.label("Probe Table 2")
    var_sc.set("table", "tbl2")
    var_ext.set("table", "tbl2")
    point_ot.set("table", "tbl2")


def create_new_model(client: mph.Client, parameters: ModelParameters) -> mph.Model:
    """Create a new model for calculating scattered field"""
    model = client.create(
        f"comsol_helholtz_sw_scattering({datetime.now().strftime('%Y-%m-%d_%H-%M-%S')})"
    )
    _prepare_model(model=model.java, parameters=parameters)
    return model


def get_results(
    model, sel_line: str = "sel_line", data: str = "dset1"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts z, freq, and complex pressure fields (p_s, p_b) from a model along the specified line selection."""

    # Create a temporary numerical evaluation object
    eval = model.java.result().numerical().create("eval_tmp", "Eval")
    eval.selection().named(sel_line)
    eval.set("data", data)
    eval.set("expr", ["z", "acpr.p_s", "ta.p_s", "acpr.p_b", "ta.p_b"])

    # Get real and imaginary parts of the results
    results_real = np.array(eval.getData())
    results_imag = np.array(eval.getImagData())

    # Coordinate along the line
    z = results_real[0]

    # Build complex quantities
    _acpr_p_s = results_real[1] + 1j * results_imag[1]
    _ta_p_s = results_real[2] + 1j * results_imag[2]
    _acpr_p_b = results_real[3] + 1j * results_imag[3]
    _ta_p_b = results_real[4] + 1j * results_imag[4]

    # Combine ACPR and TA fields (handle NaN values)
    p_s = np.where(np.isnan(_acpr_p_s), _ta_p_s, _acpr_p_s)
    p_b = np.where(np.isnan(_acpr_p_b), _ta_p_b, _acpr_p_b)

    # Get frequency array from model parameters
    freq = np.array(model.evaluate("freq"))

    # Clean up the temporary numerical object
    model.java.result().numerical().remove("eval_tmp")

    return z, freq, p_s, p_b
