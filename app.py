from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict


import folium
from folium.plugins import Draw
import pandas as pd
import shapely.geometry
import shapely.wkt
import streamlit as st
from streamlit_folium import st_folium


from services import (
    fetch_fema_flood_hazard,
    fetch_noaa_precipitation,
    fetch_usda_soil_report,
    fetch_usgs_topography,
    generate_50pct_buffer,
    identify_jurisdiction,
)


# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION (100% ENGLISH)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Civilocity — Land Development Site Due Diligence",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Increase padding-top to 3.8rem to prevent Streamlit top navigation bar from clipping the logo
st.markdown(
    """
    <meta name="robots" content="noindex, nofollow">
    <style>
        .block-container { 
            padding-top: 3.8rem !important; 
            padding-bottom: 2rem; 
        }
        .logo-wrapper {
            margin-top: 0.6rem;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# LOGO & BRANDING HEADER (CIVILOCITY)
# -----------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_candidates = [
    os.path.join(base_dir, "logo.png"),
    os.path.join(base_dir, "logo.svg"),
    os.path.join(base_dir, "logo.jpg"),
    os.path.join(base_dir, "logo.jpeg"),
    "logo.png",
    "logo.svg",
    "logo.jpg",
]


logo_file = next((f for f in logo_candidates if os.path.exists(f)), None)


if logo_file:
    col_l, col_t = st.columns([1, 3])
    with col_l:
        st.write("")  # Clean vertical spacing
        st.image(logo_file, width=250)
    with col_t:
        st.title("Land Development Site Due Diligence")
        st.caption("Automated preliminary civil engineering due diligence: NOAA Precipitation, USGS Topography, USDA Soil Survey, FEMA Flood Hazard, and Local Jurisdictions.")
else:
    # Built-in Civilocity vector branding banner (safely padded)
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin-top: 0.5rem; margin-bottom: 0.8rem; padding-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <svg width="230" height="48" viewBox="0 0 240 50" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 28 C9 24 9 18 13 14 C17 10 24 10 28 14 C30 16 31 18 31 20" stroke="#00C853" stroke-width="6.5" stroke-linecap="round"/>
                    <path d="M12 22 C15 26 15 32 19 36 C23 40 30 40 34 36 C36 34 37 32 37 30" stroke="#0084FF" stroke-width="6.5" stroke-linecap="round"/>
                    <text x="46" y="33" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="24" font-weight="900" fill="#000000" letter-spacing="1.5">CIVILOCITY</text>
                </svg>
            </div>
            <div>
                <h2 style="margin: 0; font-size: 1.5rem; color: #003366; font-weight: 700;">Land Development Due Diligence</h2>
                <p style="margin: 0; color: #64748b; font-size: 0.88rem;">Automated site analysis for US civil engineering & drainage permitting</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# -----------------------------------------------------------------------------
# ASYNCHRONOUS ENGINE DISPATCH (5 MODULES)
# -----------------------------------------------------------------------------
def execute_due_diligence(
    site_wgs84: shapely.geometry.base.BaseGeometry,
    buffered_wgs84: shapely.geometry.base.BaseGeometry,
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    wkt_poly = shapely.wkt.dumps(site_wgs84)
    tasks = {
        "noaa": (fetch_noaa_precipitation, (lat, lon)),
        "usgs": (fetch_usgs_topography, (buffered_wgs84,)),
        "usda": (fetch_usda_soil_report, (wkt_poly, site_wgs84)),
        "fema": (fetch_fema_flood_hazard, (site_wgs84, buffered_wgs84)),
        "jurisdiction": (identify_jurisdiction, (lat, lon)),
    }


    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {k: executor.submit(fn, *args) for k, (fn, args) in tasks.items()}
        for k, fut in futures.items():
            try:
                results[k] = fut.result()
            except Exception as e:
                results[k] = {
                    "status": "error",
                    "message": f"Information not found or unavailable for this location: {e}",
                }
    return results




# -----------------------------------------------------------------------------
# CARTOGRAPHIC VIEWER & DRAW CONTROLS
# -----------------------------------------------------------------------------
col_map, col_ctrl = st.columns([7, 3])


with col_map:
    m = folium.Map(location=[28.5383, -81.3792], zoom_start=14, tiles=None, control_scale=True)


    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Satellite Imagery",
        max_zoom=19,
    ).add_to(m)


    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Boundaries",
        name="Roads & Labels",
        overlay=True,
    ).add_to(m)


    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)


    if "site_geom" in st.session_state and "buffered_geom" in st.session_state:
        folium.GeoJson(
            shapely.geometry.mapping(st.session_state["buffered_geom"]),
            name="Topography Study Buffer (+50% Area)",
            style_function=lambda x: {
                "color": "#e65100",
                "weight": 2,
                "dashArray": "5, 5",
                "fillColor": "#ff9800",
                "fillOpacity": 0.15,
            },
        ).add_to(m)


        folium.GeoJson(
            shapely.geometry.mapping(st.session_state["site_geom"]),
            name="Site Boundary",
            style_function=lambda x: {
                "color": "#0091ea",
                "weight": 3,
                "fillColor": "#00b0ff",
                "fillOpacity": 0.25,
            },
        ).add_to(m)


    folium.LayerControl(position="topright").add_to(m)
    map_data = st_folium(m, height=480, width="100%", returned_objects=["last_active_drawing"])


active_drawing = map_data.get("last_active_drawing") if map_data else None


if active_drawing and active_drawing.get("geometry"):
    geom_dict = active_drawing["geometry"]
    raw_geom = shapely.geometry.shape(geom_dict)
    buffer_geom, buffer_dist = generate_50pct_buffer(raw_geom)
    cent = raw_geom.centroid


    st.session_state["site_geom"] = raw_geom
    st.session_state["buffered_geom"] = buffer_geom
    st.session_state["buffer_dist_m"] = buffer_dist
    st.session_state["centroid"] = (cent.y, cent.x)


with col_ctrl:
    st.subheader("Site Specifications")
    if "site_geom" in st.session_state:
        lat, lon = st.session_state["centroid"]
        site_m2 = st.session_state["site_geom"].area * 1e6
        buffer_dist_ft = st.session_state.get("buffer_dist_m", 30.0) * 3.28084
        topo_m2 = site_m2 * 1.50


        st.markdown(f"**Centroid:** `{lat:.6f}°, {lon:.6f}°`")
        st.markdown(f"**Site Area:** `{site_m2:.1f} m²`")
        st.markdown(f"**Topography Study Area:** `{topo_m2:.1f} m²` *(+50% Area / ~{buffer_dist_ft:.1f} ft offset)*")


        if st.button("🚀 Run Site Analysis", type="primary", use_container_width=True):
            with st.spinner("Executing NOAA, USGS, USDA, FEMA, and Census services in parallel..."):
                results = execute_due_diligence(
                    st.session_state["site_geom"],
                    st.session_state["buffered_geom"],
                    lat,
                    lon,
                )
                st.session_state["analysis_results"] = results
                st.rerun()
    else:
        st.info("Use the polygon tool on the upper-left of the map to draw the project boundary.")


# -----------------------------------------------------------------------------
# RESULTS PRESENTATION (ALL ENGLISH DUAL-ACTION CARDS)
# -----------------------------------------------------------------------------
if "analysis_results" in st.session_state:
    res = st.session_state["analysis_results"]
    st.divider()
    st.subheader("Preliminary Due Diligence Deliverables")


    col_r1, col_r2 = st.columns(2)


    # 1. NOAA ATLAS 14
    with col_r1:
        with st.container(border=True):
            st.markdown("#### 1. Precipitation Frequency (NOAA Atlas 14)")
            noaa = res.get("noaa", {})
            if noaa.get("status") == "success":
                summ = noaa.get("summary", {})
                st.write(f"**25-yr / 24-hr:** `{summ.get('25yr_24hr')}\"` | **100-yr / 24-hr:** `{summ.get('100yr_24hr')}\"`")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        label="📥 Download NOAA PDF",
                        data=noaa["pdf_bytes"],
                        file_name=noaa["filename"],
                        mime="application/pdf",
                        key="dl_noaa",
                        use_container_width=True,
                    )
                with c2:
                    st.link_button(
                        "🌐 Open NOAA PFDS Map",
                        noaa.get("noaa_url", "https://hdsc.nws.noaa.gov/pfds/"),
                        use_container_width=True,
                    )
            else:
                st.warning(f"⚠️ {noaa.get('message')}")


    # 2. USGS 3DEP TOPOGRAPHY (CON BUFFER +50% ÁREA)
    with col_r2:
        with st.container(border=True):
            st.markdown("#### 2. Site Topography (USGS 3DEP DEM)")
            usgs = res.get("usgs", {})
            if usgs.get("status") == "success":
                st.write(f"**Resolution:** `{usgs.get('dimensions')}` | **Size:** `{usgs.get('size_kb')} KB` *(+50% Area Buffer)*")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        label="📥 Download GeoTIFF (.tif)",
                        data=usgs["tiff_bytes"],
                        file_name=usgs["filename"],
                        mime="image/tiff",
                        key="dl_usgs",
                        use_container_width=True,
                    )
                with c2:
                    st.link_button(
                        "🌐 USGS National Map",
                        usgs.get("tnm_url", "https://apps.nationalmap.gov/downloader/"),
                        use_container_width=True,
                    )
            else:
                st.warning(f"⚠️ {usgs.get('message')}")


    # 3. USDA WEB SOIL SURVEY (MAP UNIT LEGEND TABLE IN APP - MAP IN PDF ONLY)
    with st.container(border=True):
        st.markdown("#### 3. Soil Resource Report (USDA NRCS SSURGO)")
        usda = res.get("usda", {})
        if usda.get("status") == "success":
            st.markdown("##### Map Unit Legend")
            units = usda.get("map_units", [])
            if units:
                table_display = []
                for u in units:
                    table_display.append({
                        "Symbol": u.get("musym"),
                        "Map Unit Name": u.get("muname"),
                        "Acres in AOI": f"{u.get('acres', 0.0):.1f}",
                        "Percent of AOI": u.get("percent", "0.0%"),
                        "HSG": u.get("hsg", "Not Rated"),
                        "Water Table (SHWT)": u.get("water_table", "N/A"),
                    })
                df_soils = pd.DataFrame(table_display)
                st.dataframe(df_soils, hide_index=True, use_container_width=True)
                st.caption(f"**Totals for Area of Interest:** `{usda.get('total_acres', 0.0):.1f} Acres (100.0%)`")


            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="📥 Download Soil Report & Map (PDF)",
                    data=usda["pdf_bytes"],
                    file_name=usda["filename"],
                    mime="application/pdf",
                    key="dl_usda",
                    use_container_width=True,
                )
            with c2:
                st.link_button(
                    "🌐 Open WSS (AOI Ready)",
                    usda.get("wss_url", "https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx"),
                    use_container_width=True,
                )
        else:
            st.warning(f"⚠️ {usda.get('message')}")


    col_r4, col_r5 = st.columns(2)


    # 4. FEMA NFHL FLOOD HAZARDS
    with col_r4:
        with st.container(border=True):
            st.markdown("#### 4. Flood Hazards (FEMA NFHL FIRMette)")
            fema = res.get("fema", {})
            if fema.get("status") == "success":
                st.write(f"**Flood Zone:** `{fema.get('flood_zone')}` | **FIRM Panel:** `{fema.get('firm_panel')}`")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        label="📥 Download FIRMette PDF",
                        data=fema["pdf_bytes"],
                        file_name=fema["filename"],
                        mime="application/pdf",
                        key="dl_fema",
                        use_container_width=True,
                    )
                with c2:
                    st.link_button(
                        "🌐 FEMA MSC FIRMette",
                        fema.get("msc_url", "https://msc.fema.gov/portal/search"),
                        use_container_width=True,
                    )
            else:
                st.warning(f"⚠️ {fema.get('message')}")


    # 5. LOCAL JURISDICTION & STANDARDS
    with col_r5:
        with st.container(border=True):
            st.markdown("#### 5. Local Jurisdiction & Regulatory Standards")
            jur = res.get("jurisdiction", {})
            if jur.get("status") == "success":
                st.info(f"**Jurisdiction:** {jur.get('summary')}")
                b1, b2 = st.columns(2)
                with b1:
                    st.link_button(
                        "📘 Stormwater Manual",
                        jur.get("stormwater_manual_url"),
                        use_container_width=True,
                    )
                with b2:
                    st.link_button(
                        "🏛️ Municode / LDR",
                        jur.get("municode_ldr_url"),
                        use_container_width=True,
                    )
            else:
                st.warning(f"⚠️ {jur.get('message')}")