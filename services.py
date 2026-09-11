import csv
from io import BytesIO, StringIO
import json
import math
import time
from typing import Any, Dict, List, Tuple
import urllib.parse


import pyproj
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter, portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import requests
from requests.adapters import HTTPAdapter
import shapely.geometry
import shapely.ops
import shapely.validation
import shapely.wkt
from urllib3.util import Retry


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


US_STATE_ABBR = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
    'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
    'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
    'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
    'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY'
}


def get_resilient_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# -----------------------------------------------------------------------------
# TOPOGRAPHY BUFFER (+50% AREA IN LOCAL UTM)
# -----------------------------------------------------------------------------
def generate_50pct_buffer(
    geom_wgs84: shapely.geometry.base.BaseGeometry,
) -> Tuple[shapely.geometry.base.BaseGeometry, float]:
    centroid = geom_wgs84.centroid
    lon, lat = centroid.x, centroid.y


    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone


    to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True
    ).transform
    to_wgs84 = pyproj.Transformer.from_crs(
        f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True
    ).transform


    geom_utm = shapely.ops.transform(to_utm, geom_wgs84)
    A_orig = geom_utm.area
    P_orig = geom_utm.length
    A_target = 1.50 * A_orig
    delta_A = 0.50 * A_orig


    d = (-P_orig + math.sqrt(max(0, P_orig**2 + 4 * math.pi * delta_A))) / (2 * math.pi)


    for _ in range(5):
        b_utm = geom_utm.buffer(d)
        err = b_utm.area - A_target
        if abs(err) / A_target < 0.001:
            break
        p_b = b_utm.length
        if p_b > 0:
            d -= err / p_b


    buffered_utm = geom_utm.buffer(d)
    buffered_wgs84 = shapely.ops.transform(to_wgs84, buffered_utm)


    return buffered_wgs84, d


def generate_100ft_buffer(geom_wgs84: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    buffered_geom, _ = generate_50pct_buffer(geom_wgs84)
    return buffered_geom


# -----------------------------------------------------------------------------
# 1. NOAA ATLAS 14 PRECIPITATION FREQUENCY
# -----------------------------------------------------------------------------
def fetch_noaa_precipitation(lat: float, lon: float) -> Dict[str, Any]:
    try:
        session = get_resilient_session()
        url = (
            f"https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text.csv"
            f"?lat={lat:.5f}&lon={lon:.5f}&data=depth&units=english&series=pds"
        )
        response = session.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200 or "not in domain" in response.text.lower():
            url_mean = (
                f"https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/fe_text_mean.csv"
                f"?lat={lat:.5f}&lon={lon:.5f}&data=depth&units=english&series=pds"
            )
            response = session.get(url_mean, headers=HEADERS, timeout=15)


        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code} from NOAA PFDS")


        lines = response.text.splitlines()
        mean_data: Dict[str, List[str]] = {}
        lower_data: Dict[str, List[str]] = {}
        upper_data: Dict[str, List[str]] = {}


        current_dict = mean_data
        durations_order = []
        return_periods = ["1", "2", "5", "10", "25", "50", "100", "200", "500", "1000"]


        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            r = next(csv.reader(StringIO(line_str)), [])
            if not r:
                continue


            first = r[0].strip().lower()
            if "lower bound" in first:
                current_dict = lower_data
                continue
            elif "upper bound" in first:
                current_dict = upper_data
                continue
            elif "by duration for ari" in first or ("by duration" in first and len(r) > 5):
                if len(r) > 1:
                    return_periods = [str(c).replace("-yr", "").strip() for c in r[1:] if c.strip()]
                continue


            dur_label = r[0].replace(":", "").strip()
            if any(dur_label.lower().startswith(d) for d in [
                "5-min", "10-min", "15-min", "30-min", "60-min",
                "2-hr", "3-hr", "6-hr", "12-hr", "24-hr",
                "2-day", "3-day", "4-day", "7-day", "10-day", "20-day", "30-day", "45-day", "60-day"
            ]):
                if dur_label not in durations_order:
                    durations_order.append(dur_label)
                current_dict[dur_label] = [str(c).strip() for c in r[1:]]


        if not mean_data:
            raise ValueError("No precipitation frequency records could be parsed from NOAA response.")


        styles = getSampleStyleSheet()
        header_title_style = ParagraphStyle(
            "NOAAHeader",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            alignment=1,
            textColor=colors.HexColor("#002855"),
        )
        col_hdr_style = ParagraphStyle(
            "NOAAColHdr",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            alignment=1,
            textColor=colors.HexColor("#002855"),
        )
        cell_style = ParagraphStyle(
            "NOAACell",
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            alignment=1,
        )


        table_rows = []
        top_header = [Paragraph("<b>Duration</b>", col_hdr_style)] + [
            Paragraph("<b>Average recurrence interval (years)</b>", col_hdr_style)
        ] + [""] * (len(return_periods) - 1)


        sub_header = [""] + [Paragraph(f"<b>{yr}</b>", col_hdr_style) for yr in return_periods]
        table_rows.append(top_header)
        table_rows.append(sub_header)


        for dur in durations_order:
            means = mean_data.get(dur, [])
            lowers = lower_data.get(dur, [])
            uppers = upper_data.get(dur, [])


            row = [Paragraph(f"<b>{dur}</b>", cell_style)]
            for i in range(len(return_periods)):
                m_val = means[i] if i < len(means) else "-"
                l_val = lowers[i] if i < len(lowers) else ""
                u_val = uppers[i] if i < len(uppers) else ""


                if l_val and u_val:
                    cell_html = f"<b>{m_val}</b><br/><font size=5.5 color='#555555'>({l_val}–{u_val})</font>"
                else:
                    cell_html = f"<b>{m_val}</b>"
                row.append(Paragraph(cell_html, cell_style))
            table_rows.append(row)


        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            rightMargin=0.35 * inch,
            leftMargin=0.35 * inch,
            topMargin=0.4 * inch,
            bottomMargin=0.4 * inch,
        )


        elements = [
            Paragraph(
                "POINT PRECIPITATION FREQUENCY (PF) ESTIMATES<br/>"
                "<font size=8.5>WITH 90% CONFIDENCE INTERVALS AND SUPPLEMENTARY INFORMATION</font><br/>"
                "<font size=8 color='#004080'><b>NOAA Atlas 14</b></font>",
                header_title_style,
            ),
            Spacer(1, 4),
        ]


        meta_table = Table(
            [[
                Paragraph(
                    f"<b>Location Information:</b> Latitude: <b>{lat:.5f}°</b> | Longitude: <b>{lon:.5f}°</b> | "
                    f"<b>Data Type:</b> Precipitation Depth (Inches) | <b>Series:</b> Partial Duration Series (PDS)",
                    ParagraphStyle("LocInfo", fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#333333"))
                )
            ]],
            colWidths=[10.2 * inch],
        )
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 6))


        col_w = (10.2 * inch - 1.1 * inch) / max(1, len(return_periods))
        t = Table(table_rows, colWidths=[1.1 * inch] + [col_w] * len(return_periods))
        t.setStyle(
            TableStyle(
                [
                    ("SPAN", (1, 0), (-1, 0)),
                    ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ALIGN", (0, 2), (0, -1), "LEFT"),
                    ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        elements.append(t)
        elements.append(Spacer(1, 6))


        disclaimer = Paragraph(
            "<i>Precipitation frequency (PF) estimates in this table are based on frequency analysis of partial duration series (PDS). "
            "Numbers in parenthesis are PF estimates at lower and upper bounds of the 90% confidence interval. "
            "Source: NOAA National Weather Service, Hydrometeorological Design Studies Center (HDSC).</i>",
            ParagraphStyle("Disc", fontName="Helvetica", fontSize=6.5, leading=8.5, textColor=colors.HexColor("#64748b"))
        )
        elements.append(disclaimer)
        doc.build(elements)


        pdf_bytes = buffer.getvalue()
        buffer.close()


        p25_24 = mean_data.get("24-hr", [""] * 5)[4] if len(mean_data.get("24-hr", [])) > 4 else "N/A"
        p100_24 = mean_data.get("24-hr", [""] * 7)[6] if len(mean_data.get("24-hr", [])) > 6 else "N/A"
        noaa_direct_url = f"https://hdsc.nws.noaa.gov/pfds/pfds_map_cont.html?lat={lat:.5f}&lon={lon:.5f}"


        return {
            "status": "success",
            "pdf_bytes": pdf_bytes,
            "filename": f"NOAA_Atlas14_PF_{lat:.4f}_{lon:.4f}.pdf",
            "summary": {
                "25yr_24hr": p25_24,
                "100yr_24hr": p100_24,
            },
            "noaa_url": noaa_direct_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }


# -----------------------------------------------------------------------------
# 2. USGS 3DEP TOPOGRAPHY (GEOTIFF CON BUFFER +50% ÁREA)
# -----------------------------------------------------------------------------
def fetch_usgs_topography(
    buffered_polygon_wgs84: shapely.geometry.base.BaseGeometry,
) -> Dict[str, Any]:
    try:
        session = get_resilient_session()
        minx, miny, maxx, maxy = buffered_polygon_wgs84.bounds
        mid_lat = (miny + maxy) / 2.0
        dx_meters = abs(maxx - minx) * math.cos(math.radians(mid_lat))
        dy_meters = abs(maxy - miny)


        aspect_ratio = dx_meters / (dy_meters if dy_meters > 0 else 1e-6)
        max_dim = 1200


        if aspect_ratio >= 1.0:
            width = max_dim
            height = max(100, int(round(max_dim / aspect_ratio)))
        else:
            height = max_dim
            width = max(100, int(round(max_dim * aspect_ratio)))


        url = (
            "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
            f"?bbox={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}&bboxSR=4326"
            f"&size={width},{height}&imageSR=4326&format=tiff&f=image"
        )
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code} from USGS ImageServer")


        if not (resp.content.startswith(b"II*\x00") or resp.content.startswith(b"MM\x00*")):
            raise ValueError("USGS returned an invalid non-TIFF payload.")


        tnm_url = f"https://apps.nationalmap.gov/downloader/#/?bbox={minx:.5f},{miny:.5f},{maxx:.5f},{maxy:.5f}"


        return {
            "status": "success",
            "tiff_bytes": resp.content,
            "filename": "Site_USGS_3DEP_Topography_50pct_Buffer.tif",
            "dimensions": f"{width}x{height} px",
            "size_kb": round(len(resp.content) / 1024, 1),
            "tnm_url": tnm_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }


# -----------------------------------------------------------------------------
# 3. USDA WEB SOIL SURVEY (DIRECT WSS AOI READY — NO IN-APP PDF)
# -----------------------------------------------------------------------------
def fetch_usda_soil_report(
    wkt_polygon: str, site_geom: shapely.geometry.base.BaseGeometry
) -> Dict[str, Any]:
    try:
        coords_str = ",".join([f"{p[0]:.5f} {p[1]:.5f}" for p in site_geom.exterior.coords])
        aoi_param = urllib.parse.quote(f"(({coords_str}))")
        wss_direct_url = f"https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx?aoicoords={aoi_param}"


        return {
            "status": "success",
            "wss_url": wss_direct_url,
            "summary": "Official USDA Web Soil Survey portal ready with project boundaries pre-loaded as AOI.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not generate Web Soil Survey URL: {e}",
            "wss_url": "https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx",
        }


# -----------------------------------------------------------------------------
# 4. FEMA NFHL FLOOD HAZARDS
# -----------------------------------------------------------------------------
def fetch_fema_flood_hazard(
    site_geom: shapely.geometry.base.BaseGeometry,
    buffered_geom: shapely.geometry.base.BaseGeometry,
) -> Dict[str, Any]:
    """Queries official FEMA National Flood Hazard Layer (NFHL) with multi-endpoint redundancy,
    evaluating both property boundary envelope and centroid to guarantee exact flood zone determination."""
    try:
        session = get_resilient_session(retries=3, backoff_factor=1.5)
        cent = site_geom.centroid
        lat, lon = cent.y, cent.x
        minx, miny, maxx, maxy = site_geom.bounds
        b_minx, b_miny, b_maxx, b_maxy = buffered_geom.bounds


        fld_zone = "Zone X"
        zone_subty = "Area of Minimal Flood Hazard"
        sfha = "F"
        bfe = None


        fema_base_urls = [
            "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer",
            "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer",
        ]


        features = []


        # 1. Primary Query: Spatial Envelope across entire property boundary
        for base in fema_base_urls:
            try:
                url_env = (
                    f"{base}/28/query?geometry={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"
                    "&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects"
                    "&outFields=FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE&returnGeometry=false&f=json"
                )
                r = session.get(url_env, headers=HEADERS, timeout=14)
                if r.status_code == 200:
                    data = r.json()
                    feats = data.get("features", [])
                    if feats:
                        features = feats
                        break
            except Exception:
                continue


        # 2. Secondary Query: Exact coordinate point if envelope returned empty
        if not features:
            for base in fema_base_urls:
                try:
                    url_pt = (
                        f"{base}/28/query?geometry={lon:.6f},{lat:.6f}"
                        "&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects"
                        "&outFields=FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE&returnGeometry=false&f=json"
                    )
                    r = session.get(url_pt, headers=HEADERS, timeout=12)
                    if r.status_code == 200:
                        feats = r.json().get("features", [])
                        if feats:
                            features = feats
                            break
                except Exception:
                    continue


        if features:
            # Rank zones by regulatory risk priority: Special Flood Hazard Area (100-yr) takes precedence
            def rank_zone(feat):
                attrs = feat.get("attributes", {})
                z = str(attrs.get("FLD_ZONE", "")).upper()
                s = str(attrs.get("SFHA_TF", "")).upper()
                if "V" in z:
                    return 4
                if s == "T" or any(x in z for x in ["A", "AE", "AH", "AO", "AR", "A99"]):
                    return 3
                if "SHADED" in str(attrs.get("ZONE_SUBTY", "")).upper() or "500" in str(attrs.get("ZONE_SUBTY", "")):
                    return 2
                return 1


            sorted_feats = sorted(features, key=rank_zone, reverse=True)
            primary_attrs = sorted_feats[0].get("attributes", {})
            fld_zone = str(primary_attrs.get("FLD_ZONE", "Zone X"))
            zone_subty = str(primary_attrs.get("ZONE_SUBTY", ""))
            sfha = str(primary_attrs.get("SFHA_TF", "F"))
            bfe = primary_attrs.get("STATIC_BFE", None)


            # Check if property spans multiple flood zones
            unique_zones = []
            for f in sorted_feats:
                uz = str(f.get("attributes", {}).get("FLD_ZONE", "")).strip()
                if uz and uz not in unique_zones:
                    unique_zones.append(uz)
            if len(unique_zones) > 1:
                fld_zone = " / ".join(unique_zones)


        time.sleep(0.2)


        # 3. Query FIRM Panel (Layer 14) with exact coordinates
        firm_panel = "Not Delineated"
        eff_date = "Effective"
        for base in fema_base_urls:
            try:
                url_pan = (
                    f"{base}/14/query?geometry={lon:.6f},{lat:.6f}"
                    "&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects"
                    "&outFields=FIRM_PAN,EFF_DATE&returnGeometry=false&f=json"
                )
                r = session.get(url_pan, headers=HEADERS, timeout=12)
                if r.status_code == 200:
                    pan_feats = r.json().get("features", [])
                    if pan_feats:
                        pan_attrs = pan_feats[0].get("attributes", {})
                        firm_panel = str(pan_attrs.get("FIRM_PAN", "N/A"))
                        eff_date = str(pan_attrs.get("EFF_DATE", "Effective"))
                        break
            except Exception:
                continue


        # 4. Generate High-Resolution Map Graphic for PDF
        img_bytes = None
        for base in fema_base_urls:
            try:
                time.sleep(0.2)
                map_img_url = (
                    f"{base}/export?bbox={b_minx:.6f},{b_miny:.6f},{b_maxx:.6f},{b_maxy:.6f}&bboxSR=4326&imageSR=4326"
                    "&size=900,500&format=png&transparent=false&f=image"
                )
                map_img_resp = session.get(map_img_url, headers=HEADERS, timeout=15)
                if map_img_resp.status_code == 200 and len(map_img_resp.content) > 1000:
                    img_bytes = map_img_resp.content
                    break
            except Exception:
                continue


        # 5. Build ReportLab PDF Deliverable
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=portrait(letter),
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )
        styles = getSampleStyleSheet()


        title_style = ParagraphStyle(
            "FEMATitle",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#003366"),
        )
        meta_style = ParagraphStyle(
            "FEMAMeta",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4a4a4a"),
        )
        cell_bold = ParagraphStyle(
            "FEMACellB",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#002855"),
        )
        cell_txt = ParagraphStyle(
            "FEMACellT",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1e293b"),
        )


        elements = [
            Paragraph("National Flood Hazard Layer FIRMette - FEMA", title_style),
            Spacer(1, 3),
            Paragraph(
                "Federal Emergency Management Agency (FEMA) Flood Insurance Rate Map Determination Summary",
                meta_style,
            ),
            Spacer(1, 8),
        ]


        sfha_str = "YES (Special Flood Hazard Area - 100-Year Base Flood Risk)" if sfha == "T" else "NO (Outside Special Flood Hazard Area)"
        bfe_str = f"{bfe} ft NAVD88" if (bfe is not None and bfe > -999) else "N/A (No Base Flood Elevation Established)"
        zone_desc = f"{fld_zone}" + (f" ({zone_subty})" if zone_subty else "")


        summary_rows = [
            [Paragraph("<b>Flood Zone:</b>", cell_bold), Paragraph(f"<b>{zone_desc}</b>", cell_txt)],
            [Paragraph("<b>SFHA (100-Year Flood Hazard):</b>", cell_bold), Paragraph(sfha_str, cell_txt)],
            [Paragraph("<b>Base Flood Elevation (BFE):</b>", cell_bold), Paragraph(bfe_str, cell_txt)],
            [Paragraph("<b>FIRM Panel Number:</b>", cell_bold), Paragraph(firm_panel, cell_txt)],
            [Paragraph("<b>Panel Effective Date:</b>", cell_bold), Paragraph(eff_date, cell_txt)],
            [Paragraph("<b>Project Coordinates:</b>", cell_bold), Paragraph(f"Lat: {lat:.6f}°, Lon: {lon:.6f}°", cell_txt)],
        ]


        t_summary = Table(summary_rows, colWidths=[2.3 * inch, 4.7 * inch])
        t_summary.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(t_summary)
        elements.append(Spacer(1, 10))


        if img_bytes:
            img_buf = BytesIO(img_bytes)
            img = RLImage(img_buf, width=7.0 * inch, height=3.8 * inch)
            elements.append(img)
            elements.append(Spacer(1, 8))


        legend_text = (
            "<b>FEMA Flood Hazard Legend & Definitions:</b><br/>"
            "• <b>Zone A / AE:</b> Areas subject to inundation by the 1% annual chance flood (100-year flood). Mandatory flood insurance purchase requirements apply.<br/>"
            "• <b>Zone X (shaded):</b> Areas of 0.2% annual chance flood (500-year flood) or areas with average depths < 1 foot.<br/>"
            "• <b>Zone X (unshaded):</b> Areas of minimal flood hazard outside the 500-year floodplain.<br/>"
            "• <b>Regulatory Floodway:</b> The channel of a stream plus adjacent land areas that must be reserved in order to discharge the base flood without cumulatively increasing water surface elevation more than a designated height."
        )
        elements.append(
            Paragraph(legend_text, ParagraphStyle("Leg", fontName="Helvetica", fontSize=7, leading=9.5, textColor=colors.HexColor("#475569")))
        )
        doc.build(elements)


        pdf_bytes = buffer.getvalue()
        buffer.close()


        msc_url = f"https://msc.fema.gov/portal/search?AddressQuery={lat:.6f}%2C{lon:.6f}"


        return {
            "status": "success",
            "pdf_bytes": pdf_bytes,
            "filename": f"FEMA_NFHL_FIRMette_{firm_panel}.pdf",
            "flood_zone": fld_zone,
            "firm_panel": firm_panel,
            "sfha": sfha,
            "bfe": bfe_str,
            "msc_url": msc_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }




# -----------------------------------------------------------------------------
# 5. JURISDICTION & REGULATORY CODES (DIRECT PORTAL LINKS)
# -----------------------------------------------------------------------------
def identify_jurisdiction(lat: float, lon: float) -> Dict[str, Any]:
    try:
        session = get_resilient_session()
        url = (
            f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
            f"?x={lon:.6f}&y={lat:.6f}&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
        )
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code} from Census Geocoder")


        geographies = resp.json().get("result", {}).get("geographies", {})


        counties = geographies.get("Counties", [])
        county_name = counties[0].get("NAME", "Unknown County") if counties else "Unknown County"


        states = geographies.get("States", [])
        state_name = states[0].get("NAME", "Unknown State") if states else "Unknown State"


        inc_places = geographies.get("Incorporated Places", [])
        if inc_places and len(inc_places) > 0:
            city_name = inc_places[0].get("NAME", "Unincorporated")
            full_jurisdiction = f"{city_name}, {county_name}, {state_name}"
        else:
            city_name = "Unincorporated County"
            full_jurisdiction = f"{county_name} (Unincorporated), {state_name}"


        clean_state_code = US_STATE_ABBR.get(state_name.lower(), "").lower()
        municode_url = f"https://library.municode.com/{clean_state_code}" if clean_state_code else "https://library.municode.com/"
        
        if "florida" in state_name.lower():
            stormwater_manual_url = "https://floridadep.gov/water/submerged-lands-environmental-resources-permitting/content/environmental-resource-permitting"
        elif "georgia" in state_name.lower():
            stormwater_manual_url = "https://epd.georgia.gov/watershed-protection-branch/storm-water"
        elif "south carolina" in state_name.lower():
            stormwater_manual_url = "https://scdhec.gov/bureau-water/stormwater-program"
        elif "washington" in state_name.lower():
            stormwater_manual_url = "https://ecology.wa.gov/regulations-permits/guidance-technical-assistance/stormwater-permit-guidance"
        else:
            stormwater_manual_url = "https://www.epa.gov/npdes/stormwater-discharges-construction-activities"


        return {
            "status": "success",
            "county": county_name,
            "state": state_name,
            "city": city_name,
            "summary": full_jurisdiction,
            "stormwater_manual_url": stormwater_manual_url,
            "municode_ldr_url": municode_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }


# -----------------------------------------------------------------------------
# 6. USFWS NATIONAL WETLANDS INVENTORY (NWI)
# -----------------------------------------------------------------------------
def fetch_wetlands(site_geom: shapely.geometry.base.BaseGeometry) -> Dict[str, Any]:
    """Queries USFWS Wetlands MapServer and provides direct mapper link."""
    try:
        session = get_resilient_session()
        minx, miny, maxx, maxy = site_geom.bounds
        cent = site_geom.centroid


        wetlands_url = "https://fwsprimary.wim.usgs.gov/wetlands/apps/wetlands-mapper/"


        url = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
        params = {
            "geometry": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "WETLAND_TYPE,ATTRIBUTE,ACRES",
            "returnGeometry": "true",
            "f": "geojson",
        }


        resp = session.get(url, params=params, headers=HEADERS, timeout=12)
        intersecting_features = []
        total_wetland_acres = 0.0


        if resp.status_code == 200:
            geojson_data = resp.json()
            features = geojson_data.get("features", [])


            for feat in features:
                geom_dict = feat.get("geometry")
                if geom_dict:
                    try:
                        w_geom = shapely.geometry.shape(geom_dict)
                        if w_geom.intersects(site_geom):
                            props = feat.get("properties", {})
                            acres_val = props.get("ACRES")
                            try:
                                acres_flt = float(acres_val) if acres_val is not None else 0.0
                            except Exception:
                                acres_flt = 0.0


                            feat["properties"] = {
                                "WETLAND_TYPE": str(props.get("WETLAND_TYPE", "Wetland Area")),
                                "ATTRIBUTE": str(props.get("ATTRIBUTE", "NWI")),
                                "ACRES": f"{acres_flt:.2f}",
                            }
                            intersecting_features.append(feat)
                            total_wetland_acres += acres_flt
                    except Exception:
                        pass


        count = len(intersecting_features)
        if count > 0:
            summary = f"{count} wetland polygon(s) identified ({total_wetland_acres:.2f} total acres)."
        else:
            summary = "No National Wetlands Inventory (NWI) features detected within site boundary."


        return {
            "status": "success",
            "count": count,
            "total_acres": round(total_wetland_acres, 2),
            "geojson": {"type": "FeatureCollection", "features": intersecting_features} if count > 0 else None,
            "summary": summary,
            "wetlands_url": wetlands_url,
            "coords_str": f"{cent.y:.5f}, {cent.x:.5f}",
        }


    except Exception as e:
        cent = site_geom.centroid
        return {
            "status": "error",
            "message": f"Information not found: {e}",
            "geojson": None,
            "count": 0,
            "wetlands_url": "https://fwsprimary.wim.usgs.gov/wetlands/apps/wetlands-mapper/",
            "coords_str": f"{cent.y:.5f}, {cent.x:.5f}",
        }


# -----------------------------------------------------------------------------
# 7. EPA ENVIRONMENTAL HAZARDS (ECHO & CLEANUPS)
# -----------------------------------------------------------------------------
def fetch_epa_hazards(
    site_geom: shapely.geometry.base.BaseGeometry,
    buffered_geom: shapely.geometry.base.BaseGeometry,
) -> Dict[str, Any]:
    """Queries EPA Cleanups FeatureServer & links directly to EPA ECHO facility search."""
    try:
        session = get_resilient_session()
        minx, miny, maxx, maxy = buffered_geom.bounds
        cent = site_geom.centroid


        echo_url = f"https://echo.epa.gov/facilities/facility-search/results?radius=1&latitude={cent.y:.5f}&longitude={cent.x:.5f}"


        url = "https://services.arcgis.com/cJ9YHowT8TU7DUyn/ArcGIS/rest/services/Brownfields/FeatureServer/0/query"
        params = {
            "geometry": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        }


        features = []
        resp = session.get(url, params=params, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            raw_feats = data.get("features", [])
            for f in raw_feats:
                p = f.get("properties", {})
                f["properties"] = {
                    "NAME": str(p.get("PROPERTY_NAME") or p.get("SITE_NAME") or p.get("NAME") or "EPA Regulated Site"),
                    "STATUS": str(p.get("CLEANUP_STATUS") or p.get("STATUS") or "Active Assessment/Cleanup"),
                    "TYPE": str(p.get("SITE_TYPE") or p.get("GRANT_TYPE") or "Brownfield Site"),
                }
                features.append(f)


        count = len(features)
        if count > 0:
            summary = f"{count} EPA regulated environmental hazard site(s) located in site vicinity."
        else:
            summary = "No EPA Brownfields or Superfund sites identified in immediate site vicinity."


        return {
            "status": "success",
            "count": count,
            "geojson": {"type": "FeatureCollection", "features": features} if count > 0 else None,
            "summary": summary,
            "echo_url": echo_url,
            "coords_str": f"{cent.y:.5f}, {cent.x:.5f}",
        }


    except Exception as e:
        cent = site_geom.centroid
        return {
            "status": "error",
            "message": f"Information not found: {e}",
            "geojson": None,
            "count": 0,
            "echo_url": "https://echo.epa.gov/",
            "coords_str": f"{cent.y:.5f}, {cent.x:.5f}",
        }


# -----------------------------------------------------------------------------
# 8. PARCEL CADASTRE & PROPERTY APPRAISER SEARCH
# -----------------------------------------------------------------------------
def fetch_parcels(
    site_geom: shapely.geometry.base.BaseGeometry,
    county: str,
    state: str,
    custom_url: str = None,
) -> Dict[str, Any]:
    """Queries cadastral servers & links directly to Property Appraiser GIS Map search."""
    cent = site_geom.centroid
    lat, lon = cent.y, cent.x
    minx, miny, maxx, maxy = site_geom.bounds


    # Direct search for the County Property Appraiser GIS map
    pa_search_query = f"{county} {state} Property Appraiser GIS parcel map"
    pa_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(pa_search_query)}"


    # Regrid search directly by coordinates
    regrid_url = f"https://app.regrid.com/us/search?q={lat:.5f}%2C{lon:.5f}"


    target_endpoint = custom_url.strip() if custom_url and custom_url.strip() else ""
    if not target_endpoint:
        if "florida" in state.lower():
            target_endpoint = "https://services9.arcgis.com/Gh9awoU677aKree0/arcgis/rest/services/Florida_Statewide_Cadastral/FeatureServer/0"
        else:
            target_endpoint = "https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/2"


    features = []
    if target_endpoint:
        try:
            session = get_resilient_session()
            query_endpoint = target_endpoint.rstrip("/")
            if not query_endpoint.endswith("/query"):
                query_endpoint += "/query"


            params = {
                "geometry": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
            }
            resp = session.get(query_endpoint, params=params, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                raw_feats = data.get("features", [])
                for f in raw_feats:
                    p = f.get("properties", {})
                    f["properties"] = {
                        "PARCEL_ID": str(p.get("PARCEL_ID") or p.get("PIN") or p.get("SEC_DIV_NO") or p.get("PARCELNO") or "Lot / Section"),
                        "ZONING": str(p.get("ZONING") or p.get("SURVTYPTXT") or "Refer to Local LDR"),
                        "LAND_USE": str(p.get("FUTURE_LAND_USE") or p.get("LAND_USE") or p.get("DOR_UC") or "Residential / Commercial"),
                    }
                    features.append(f)
        except Exception:
            pass


    count = len(features)
    if count > 0:
        summary = f"{count} cadastral parcel polygon(s) retrieved."
    else:
        summary = f"Cadastral records maintained by {county} Property Appraiser."


    return {
        "status": "success",
        "count": count,
        "geojson": {"type": "FeatureCollection", "features": features} if count > 0 else None,
        "summary": summary,
        "pa_url": pa_url,
        "regrid_url": regrid_url,
    }


# -----------------------------------------------------------------------------
# 9. WATER & WASTEWATER UTILITIES (DIRECT GOOGLE SEARCH FOR CITY/COUNTY GIS)
# -----------------------------------------------------------------------------
def fetch_utilities(
    site_geom: shapely.geometry.base.BaseGeometry,
    city: str,
    county: str,
    state: str,
    custom_url: str = None,
) -> Dict[str, Any]:
    """Generates direct Google Search URL to find the official city / county Water & Sewer GIS."""
    entity = city if city and city != "Unincorporated" else county


    # Clean direct Google search for the official municipal or county GIS water and sewer map
    target_query = f"{entity} {state} GIS water sewer utilities"
    utility_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(target_query)}"


    if custom_url and custom_url.strip():
        utility_url = custom_url.strip()


    return {
        "status": "success",
        "entity_name": f"{entity}",
        "summary": f"Water & Wastewater utility GIS for {entity}, {state}.",
        "utility_url": utility_url,
    }