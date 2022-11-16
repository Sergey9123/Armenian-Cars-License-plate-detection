import PIL.ExifTags
from gmplot import gmplot
from geopy.geocoders import Nominatim
import streamlit as s


def geo_position(img):
    exif = {
        PIL.ExifTags.TAGS[k]:v
        for k,v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    return (exif)
#     north = exif["GPSInfo"][2]
#     east = exif["GPSInfo"][4]
#     lat = (((north[0] * 60) * north[1] * 60)*north[2]) / 60 / 60
#     long = (((east[0] * 60) * east[1] * 60)*east[2]) / 60 / 60
#     lat,long = float(lat),float(long)
#     gmap = gmplot.GoogleMapPlotter(lat,long,12)
#     gmap.marker(lat,long,"blue")
#     gmap.draw("location.html")
#     geoLoc = Nominatim(user_agent = "GetLoc")
#     locname = geo











