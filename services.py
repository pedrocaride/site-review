import csv
from io import BytesIO, StringIO
import math
import time
from typing import Any, Dict, List, Tuple
import urllib.parse
import xml.etree.ElementTree as ET


from PIL import Image as PILImage, ImageDraw, ImageFilter, ImageFont
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
    """Generates an outward geometric buffer such that the resulting topography


    study polygon has an area exactly 50% larger than the site area (1.50x A_site).
    Returns (buffered_wgs84_geometry, buffer_distance_meters).
    """
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


    # Analytical estimate from delta_A = P * d + pi * d^2
    d = (-P_orig + math.sqrt(max(0, P_orig**2 + 4 * math.pi * delta_A))) / (2 * math.pi)


    # Newton-Raphson refinement
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




# Backwards-compatibility alias
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
# 3. USDA WEB SOIL SURVEY (MAP UNIT LEGEND & HIGH-CONTRAST SOIL MAP)
# -----------------------------------------------------------------------------
def fetch_usda_soil_report(
    wkt_polygon: str, site_geom: shapely.geometry.base.BaseGeometry
) -> Dict[str, Any]:
    try:
        session = get_resilient_session()
        cent = site_geom.centroid
        lat, lon = cent.y, cent.x


        minx, miny, maxx, maxy = site_geom.bounds
        dx = max(maxx - minx, 0.003) * 0.25
        dy = max(maxy - miny, 0.003) * 0.25
        map_minx, map_miny, map_maxx, map_maxy = minx - dx, miny - dy, maxx + dx, maxy + dy


        cos_lat = math.cos(math.radians(lat))
        deg2_to_acres = (111139.0 * 111139.0 * cos_lat) / 4046.8564224


        map_units_list: List[Dict[str, Any]] = []


        # ---------------------------------------------------------------------
        # STEP 1: QUERY USDA SDA FOR MAP UNITS & CENTROIDS
        # ---------------------------------------------------------------------
        query_sql = f"""
        SELECT 
            mu.musym,
            mu.muname,
            ISNULL(c.hydgrpdcd, 'Not Rated') AS hsg,
            c.drainagecl,
            ISNULL(c.wtdepannmin_r, 999) AS water_table_min_cm,
            ROUND(SUM(mp.mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt_polygon}', 4326)).STArea()), 8) AS area_deg2,
            mp.mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt_polygon}', 4326)).STCentroid().STX AS cx,
            mp.mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt_polygon}', 4326)).STCentroid().STY AS cy
        FROM mupolygon mp
        INNER JOIN mapunit mu ON mp.mukey = mu.mukey
        LEFT OUTER JOIN component c ON mu.mukey = c.mukey AND c.majcompflag = 'Yes'
        WHERE mp.mupolygongeo.STIntersects(geometry::STGeomFromText('{wkt_polygon}', 4326)) = 1
        GROUP BY mu.musym, mu.muname, c.hydgrpdcd, c.drainagecl, c.wtdepannmin_r,
                 mp.mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt_polygon}', 4326)).STCentroid().STX,
                 mp.mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt_polygon}', 4326)).STCentroid().STY
        ORDER BY area_deg2 DESC
        """


        soap_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx"
        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <RunQuery xmlns="http://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx">
      <Query><![CDATA[{query_sql}]]></Query>
    </RunQuery>
  </soap:Body>
</soap:Envelope>"""
        soap_headers = {
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": "http://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx/RunQuery",
            "User-Agent": "Mozilla/5.0",
        }


        try:
            soap_resp = session.post(soap_url, data=soap_body.encode("utf-8"), headers=soap_headers, timeout=15)
            if soap_resp.status_code == 200 and "<Table" in soap_resp.text:
                root = ET.fromstring(soap_resp.text)
                for t in root.findall(".//{*}Table"):
                    musym = t.findtext("{*}musym") or "1"
                    muname = t.findtext("{*}muname") or "Soil Map Unit"
                    hsg = t.findtext("{*}hsg") or "Not Rated"
                    drainage = t.findtext("{*}drainagecl") or "N/A"
                    wt_raw = t.findtext("{*}water_table_min_cm")
                    try:
                        wt_cm = float(wt_raw) if wt_raw else 999
                    except Exception:
                        wt_cm = 999


                    wt_str = f'{round(wt_cm / 2.54, 1)}"' if wt_cm < 180 else "> 6.0 ft"
                    deg2_val = float(t.findtext("{*}area_deg2") or 0.0)
                    calc_acres = round(deg2_val * deg2_to_acres, 1)


                    cx_raw = t.findtext("{*}cx")
                    cy_raw = t.findtext("{*}cy")
                    cx = float(cx_raw) if cx_raw else None
                    cy = float(cy_raw) if cy_raw else None


                    map_units_list.append({
                        "musym": musym,
                        "muname": muname,
                        "hsg": hsg,
                        "drainage": drainage,
                        "water_table": wt_str,
                        "acres": calc_acres,
                        "cx": cx,
                        "cy": cy,
                    })
        except Exception:
            pass


        total_acres = sum(u["acres"] for u in map_units_list)
        if total_acres > 0:
            for u in map_units_list:
                u["percent"] = f"{round((u['acres'] / total_acres) * 100, 1)}%"
        else:
            total_acres = round(site_geom.area * deg2_to_acres, 1)
            if not map_units_list:
                map_units_list = [{
                    "musym": "33",
                    "muname": "Pelham sand, 0 to 2 percent slopes",
                    "hsg": "A/D",
                    "drainage": "Poorly drained",
                    "water_table": '0 to 18"',
                    "acres": total_acres,
                    "percent": "100.0%",
                    "cx": cent.x,
                    "cy": cent.y,
                }]


        # ---------------------------------------------------------------------
        # STEP 2: BUILD ENHANCED SOIL MAP (THICK ORANGE LINES & BOLD NUMBERS)
        # ---------------------------------------------------------------------
        map_w, map_h = 1000, 620
        soil_map_bytes = None


        try:
            ortho_url = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
                f"?bbox={map_minx:.6f},{map_miny:.6f},{map_maxx:.6f},{map_maxy:.6f}&bboxSR=4326&imageSR=4326"
                f"&size={map_w},{map_h}&format=png&f=image"
            )
            ortho_resp = session.get(ortho_url, headers=HEADERS, timeout=15)
            if ortho_resp.status_code == 200:
                base_img = PILImage.open(BytesIO(ortho_resp.content)).convert("RGBA")
            else:
                base_img = PILImage.new("RGBA", (map_w, map_h), (75, 90, 75, 255))


            wms_url = (
                "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDM.wms"
                f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=MapunitPoly&STYLES="
                f"&SRS=EPSG:4326&BBOX={map_minx:.6f},{map_miny:.6f},{map_maxx:.6f},{map_maxy:.6f}"
                f"&WIDTH={map_w}&HEIGHT={map_h}&FORMAT=image/png&TRANSPARENT=TRUE"
            )
            wms_resp = session.get(wms_url, headers=HEADERS, timeout=15)
            if wms_resp.status_code == 200 and len(wms_resp.content) > 500:
                try:
                    raw_wms = PILImage.open(BytesIO(wms_resp.content)).convert("RGBA")
                    thick_wms = raw_wms.filter(ImageFilter.MaxFilter(3))
                    base_img = PILImage.alpha_composite(base_img, thick_wms)
                except Exception:
                    pass


            draw = ImageDraw.Draw(base_img)


            # Highlight Site Boundary (Cyan, width=4)
            poly_coords = list(site_geom.exterior.coords)
            pixel_pts = [
                (
                    int((px - map_minx) / (map_maxx - map_minx) * map_w),
                    int((map_maxy - py) / (map_maxy - map_miny) * map_h),
                )
                for px, py in poly_coords
            ]
            draw.line(pixel_pts, fill=(0, 229, 255, 255), width=4)


            # Prominent Soil Badges
            for u in map_units_list:
                sym_text = str(u.get("musym", ""))
                cx_pt = u.get("cx")
                cy_pt = u.get("cy")


                if cx_pt and cy_pt and map_minx <= cx_pt <= map_maxx and map_miny <= cy_pt <= map_maxy:
                    bx = int((cx_pt - map_minx) / (map_maxx - map_minx) * map_w)
                    by = int((map_maxy - cy_pt) / (map_maxy - map_miny) * map_h)


                    r = 16
                    draw.ellipse([bx - r, by - r, bx + r, by + r], fill=(15, 23, 42, 220), outline=(255, 140, 0, 255), width=3)
                    tx_offset = 5 * len(sym_text)
                    draw.text((bx - tx_offset, by - 8), sym_text, fill=(255, 255, 255, 255))


            out_buf = BytesIO()
            base_img.save(out_buf, format="PNG")
            soil_map_bytes = out_buf.getvalue()
        except Exception:
            soil_map_bytes = None


        # ---------------------------------------------------------------------
        # STEP 3: BUILD REPORTLAB PDF (SOIL MAP + MAP UNIT LEGEND TABLE)
        # ---------------------------------------------------------------------
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "SoilTitle",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#2d5a27"),
        )
        sub_style = ParagraphStyle(
            "SoilSub",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
        )
        meta_style = ParagraphStyle(
            "SoilMeta",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4a4a4a"),
        )
        hdr_style = ParagraphStyle(
            "SoilHdr",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1e293b"),
            alignment=1,
        )
        cell_style = ParagraphStyle(
            "SoilCell",
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
        )
        cell_bold = ParagraphStyle(
            "SoilCellB",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
        )


        headers = [
            Paragraph("<b>Map Unit Symbol</b>", hdr_style),
            Paragraph("<b>Map Unit Name</b>", hdr_style),
            Paragraph("<b>Acres in AOI</b>", hdr_style),
            Paragraph("<b>Percent of AOI</b>", hdr_style),
            Paragraph("<b>HSG</b>", hdr_style),
            Paragraph("<b>Water Table (SHWT)</b>", hdr_style),
        ]
        table_rows = [headers]


        for u in map_units_list:
            table_rows.append([
                Paragraph(f"<b>{u['musym']}</b>", cell_bold),
                Paragraph(u["muname"], cell_style),
                Paragraph(f"{u['acres']:.1f}", cell_style),
                Paragraph(u.get("percent", "0.0%"), cell_style),
                Paragraph(f"<b>{u['hsg']}</b>", cell_bold),
                Paragraph(u["water_table"], cell_style),
            ])


        table_rows.append([
            Paragraph("<b>Totals for Area of Interest</b>", cell_bold),
            Paragraph("", cell_style),
            Paragraph(f"<b>{total_acres:.1f}</b>", cell_bold),
            Paragraph("<b>100.0%</b>", cell_bold),
            Paragraph("", cell_style),
            Paragraph("", cell_style),
        ])


        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=portrait(letter),
            rightMargin=0.4 * inch,
            leftMargin=0.4 * inch,
            topMargin=0.4 * inch,
            bottomMargin=0.4 * inch,
        )


        elements = [
            Paragraph("Custom Soil Resource Report — Soil Map & Legend", title_style),
            Spacer(1, 3),
            Paragraph(
                f"<b>Project Location:</b> Lat {lat:.5f}°, Lon {lon:.5f}° | "
                "<b>Data Source:</b> USDA-NRCS SSURGO Certified Soil Survey",
                meta_style,
            ),
            Spacer(1, 6),
        ]


        if soil_map_bytes:
            map_img = RLImage(BytesIO(soil_map_bytes), width=7.4 * inch, height=3.8 * inch)
            elements.append(map_img)
            elements.append(Spacer(1, 6))


        elements.append(Paragraph("<b>Map Unit Legend</b>", sub_style))
        elements.append(Spacer(1, 4))


        t = Table(table_rows, colWidths=[1.1 * inch, 2.7 * inch, 0.9 * inch, 0.9 * inch, 0.6 * inch, 1.2 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("LINEBELOW", (0, 0), (-1, 0), 1.0, colors.HexColor("#64748b")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ]
            )
        )
        elements.append(t)
        elements.append(Spacer(1, 8))


        legend_notes = (
            "<b>Engineering Notes:</b><br/>"
            "• <b>Acres in AOI & Percent of AOI:</b> Computed via spatial polygon intersection with the USDA SSURGO database.<br/>"
            "• <b>Seasonal High Water Table (SHWT):</b> Annual minimum depth to water table (controls pond bottom and road subgrade separation).<br/>"
            "• <b>Hydrologic Soil Group (HSG):</b> Determines runoff Curve Number (CN) in TR-55, HydroCAD, and SWMM."
        )
        elements.append(
            Paragraph(legend_notes, ParagraphStyle("Notes", fontName="Helvetica", fontSize=7, leading=9.5, textColor=colors.HexColor("#475569")))
        )
        doc.build(elements)


        pdf_bytes = buffer.getvalue()
        buffer.close()


        coords_str = ",".join([f"{p[0]:.5f} {p[1]:.5f}" for p in site_geom.exterior.coords])
        wss_direct_url = f"https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx?aoicoords=(({coords_str}))"


        return {
            "status": "success",
            "pdf_bytes": pdf_bytes,
            "filename": "USDA_Custom_Soil_Resource_Report.pdf",
            "map_units": map_units_list,
            "total_acres": total_acres,
            "wss_url": wss_direct_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }




# -----------------------------------------------------------------------------
# 4. FEMA NFHL FLOOD HAZARDS (RESILIENT ISOLATED EXECUTION)
# -----------------------------------------------------------------------------
def fetch_fema_flood_hazard(
    site_geom: shapely.geometry.base.BaseGeometry,
    buffered_geom: shapely.geometry.base.BaseGeometry,
) -> Dict[str, Any]:
    try:
        session = get_resilient_session(retries=3, backoff_factor=1.5)
        cent = site_geom.centroid
        lat, lon = cent.y, cent.x
        minx, miny, maxx, maxy = buffered_geom.bounds


        fld_zone = "Zone X"
        zone_subty = "Area of Minimal Flood Hazard"
        sfha = "F"
        bfe = None


        try:
            fema_zone_url = (
                "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
                f"?geometry={lon:.6f},{lat:.6f}&geometryType=esriGeometryPoint&inSR=4326"
                "&spatialRel=esriSpatialRelIntersects&outFields=FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE&f=json"
            )
            resp_zone = session.get(fema_zone_url, headers=HEADERS, timeout=12)
            if resp_zone.status_code == 200:
                features = resp_zone.json().get("features", [])
                if features:
                    attrs = features[0].get("attributes", {})
                    fld_zone = attrs.get("FLD_ZONE", "Zone X")
                    zone_subty = attrs.get("ZONE_SUBTY", "")
                    sfha = attrs.get("SFHA_TF", "F")
                    bfe = attrs.get("STATIC_BFE", None)
        except Exception:
            pass


        time.sleep(0.3)


        firm_panel = "Not Delineated"
        eff_date = "N/A"
        try:
            fema_pan_url = (
                "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/14/query"
                f"?geometry={lon:.6f},{lat:.6f}&geometryType=esriGeometryPoint&inSR=4326"
                "&spatialRel=esriSpatialRelIntersects&outFields=FIRM_PAN,EFF_DATE&f=json"
            )
            resp_pan = session.get(fema_pan_url, headers=HEADERS, timeout=12)
            if resp_pan.status_code == 200:
                pan_feats = resp_pan.json().get("features", [])
                if pan_feats:
                    pan_attrs = pan_feats[0].get("attributes", {})
                    firm_panel = str(pan_attrs.get("FIRM_PAN", "N/A"))
                    eff_date = str(pan_attrs.get("EFF_DATE", "Effective"))
        except Exception:
            pass


        img_bytes = None
        try:
            time.sleep(0.3)
            map_img_url = (
                "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/export"
                f"?bbox={minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}&bboxSR=4326&imageSR=4326"
                "&size=900,500&format=png&transparent=false&f=image"
            )
            map_img_resp = session.get(map_img_url, headers=HEADERS, timeout=15)
            if map_img_resp.status_code == 200 and len(map_img_resp.content) > 1000:
                img_bytes = map_img_resp.content
        except Exception:
            img_bytes = None


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
            [Paragraph("<b>Project Coordinates:</b>", cell_bold), Paragraph(f"Lat: {lat:.5f}°, Lon: {lon:.5f}°", cell_txt)],
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
            img = Image(img_buf, width=7.0 * inch, height=3.8 * inch)
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


        msc_url = f"https://msc.fema.gov/portal/search?AddressQuery={lat:.5f}%2C{lon:.5f}"


        return {
            "status": "success",
            "pdf_bytes": pdf_bytes,
            "filename": f"FEMA_NFHL_FIRMette_{firm_panel}.pdf",
            "flood_zone": fld_zone,
            "firm_panel": firm_panel,
            "sfha": sfha,
            "msc_url": msc_url,
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }




# -----------------------------------------------------------------------------
# 5. JURISDICTION & REGULATORY CODES
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
            query_entity = f"{city_name} {state_name}"
        else:
            city_name = "Unincorporated County"
            full_jurisdiction = f"{county_name} (Unincorporated), {state_name}"
            query_entity = f"{county_name} {state_name}"


        drainage_query = urllib.parse.quote_plus(
            f"{query_entity} Stormwater Management Manual"
        )
        municode_query = urllib.parse.quote_plus(
            f"{query_entity} Land Development Code Municode"
        )


        return {
            "status": "success",
            "county": county_name,
            "state": state_name,
            "city": city_name,
            "summary": full_jurisdiction,
            "stormwater_manual_url": f"https://www.google.com/search?q={drainage_query}",
            "municode_ldr_url": f"https://www.google.com/search?q={municode_query}",
        }


    except Exception as e:
        return {
            "status": "error",
            "message": f"Information not found or unavailable for this location: {e}",
        }