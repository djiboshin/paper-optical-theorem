import mph
import jpype
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelParameters:
    a: str = "1[m]"
    c_host: str = "1[m/s]"
    c_sphere: str = "0.3[m/s]"
    rho_host: str = "1[kg/m^3]"
    rho_sphere: str = "1[kg/m^3]"
    p0: str = "1[Pa]"
    R_PML: str = "25[m]"
    L_PML: str = "5[m]"
    R_int: str = "8[m]"
    mesh_h: str = "0.5[m]"
    freq_start: str = "0.001[Hz]"
    freq_step: str = "0.003[Hz]"
    freq_stop: str = "0.3[Hz]"


def _prepare_model(model, parameters: ModelParameters):
    """Prepeare physics, geometry, variables, study in the model"""
    # set model parameters
    params = model.param()

    params.set("L_PML", parameters.L_PML)
    params.set("R_PML", parameters.R_PML)
    params.set("a", parameters.a)
    params.set("c_host", parameters.c_host)
    params.set("c_sphere", parameters.c_sphere)
    params.set("rho_host", parameters.rho_host)
    params.set("rho_sphere", parameters.rho_sphere)
    params.set("p0", parameters.p0)
    params.set("R_int", parameters.R_int)
    params.set("mesh_h", parameters.mesh_h)

    # create a component with geometry

    comp1 = model.component().create("comp1", True)
    geom1 = comp1.geom().create("geom1", 2)
    geom1.axisymmetric(True)

    # create physics
    acpr = comp1.physics().create("acpr", "PressureAcoustics", geom1.tag())

    # create circle for host
    c1 = geom1.create("c1", "Circle")
    c1.label("host")
    c1.set("layername", "Layer 1")
    c1.setIndex("layer", "L_PML", 0)
    c1.set("r", "R_PML+L_PML")

    # create circle for particle
    c2 = geom1.create("c2", "Circle")
    c2.label("particle")
    c2.set("r", "a")

    # create circle for integration surface
    c3 = geom1.create("c3", "Circle")
    c3.label("integration surface")
    c3.set("r", "R_int")

    # build the geometry
    geom1.run()

    # create selection for integration surface
    sel_int_surf = comp1.selection().create("sel_int_surf", "Disk")
    sel_int_surf.set("entitydim", jpype.types.JInt(1))
    sel_int_surf.label("integration surfaces")
    sel_int_surf.set("r", "1.01 * R_int")
    sel_int_surf.set("rin", "0.99 * R_int")
    sel_int_surf.set("condition", "allvertices")

    # create selection for scatterer
    sel_scatterer = comp1.selection().create("sel_scatterer", "Disk")
    sel_scatterer.label("scatterer domain")
    sel_scatterer.set("r", "0.99 * R_int")
    sel_scatterer.set("condition", "inside")

    # create selection for PML domain
    sel_PML = comp1.selection().create("sel_PML", "Disk")
    sel_PML.label("PML domains")
    sel_PML.set("r", "R_PML+L_PML")
    sel_PML.set("rin", "R_PML")
    sel_PML.set("condition", "allvertices")

    # create selection for how with PML domain (without scatterer)
    sel_host_with_PML = comp1.selection().create("sel_host_with_PML", "Complement")
    sel_host_with_PML.label("host with PML")
    sel_host_with_PML.set("input", sel_scatterer.tag())

    # create PML
    pml = comp1.coordSystem().create("pml1", "PML")
    pml.selection().named(sel_PML.tag())

    # create materials
    mat_host = comp1.material().create("mat_host", "Common")
    mat_host.selection().named(sel_host_with_PML.tag())
    mat_host.label("Host material")
    mat_host.propertyGroup("def").set("density", "rho_host")
    mat_host.propertyGroup("def").set("soundspeed", "c_host")

    mat_sphere = comp1.material().create("mat_sphere", "Common")
    mat_sphere.selection().named(sel_scatterer.tag())
    mat_sphere.label("Scatterer material")
    mat_sphere.propertyGroup("def").set("density", "rho_sphere")
    mat_sphere.propertyGroup("def").set("soundspeed", "c_sphere")

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

    # prepare physics
    acpr.prop("cref").set("cref", "c_host")

    # create background pressure field
    bpf1 = acpr.create("bpf1", "BackgroundPressureField", jpype.types.JInt(2))
    bpf1.selection().all()
    bpf1.set("pamp", "p0")
    bpf1.set("c", "c_host")

    # create mesh
    mesh1 = comp1.mesh().create("mesh1")
    mesh1.create("ftri1", "FreeTri")
    mesh1.feature("size").set("custom", "on")
    mesh1.feature("size").set("hmax", "mesh_h")
    mesh1.feature("size").set("hmin", "0.01*mesh_h")
    mesh1.run()

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


def create_new_model(client: mph.Client, parameters: ModelParameters) -> mph.Model:
    """Create a new model for calculating scattered field"""
    model = client.create(
        f"comsol_sphere_scattering({datetime.now().strftime('%Y-%m-%d_%H-%M-%S')})"
    )
    _prepare_model(model=model.java, parameters=parameters)
    return model
