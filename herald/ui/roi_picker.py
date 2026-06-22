"""ROI picker UI and errors."""


class ROIPickerError(RuntimeError):
    pass


def _build_roi_picker_ui() -> str:
    """Return Leaflet.Draw + geolocation script for rectangle ROI selection.

    Expects ``map``, ``setBanner``, ``fallbackLat``, ``fallbackLon``, and
    ``hasFallback`` to already exist in the page script scope.
    """
    return """
    let drawControl = null;
    const drawn = new L.FeatureGroup();
    map.addLayer(drawn);

    function setMapCenter(lat, lon, message) {
      map.setView([lat, lon], 17);
      setBanner(message);
    }

    function initGeolocation() {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (pos) => {
            setMapCenter(
              pos.coords.latitude,
              pos.coords.longitude,
              'Centered on your location — draw a <b>rectangle</b> over your site.'
            );
          },
          (err) => {
            if (hasFallback) {
              setMapCenter(
                fallbackLat,
                fallbackLon,
                'Using fallback center — draw a <b>rectangle</b> over your site.'
              );
            } else {
              setBanner(
                'Location unavailable (' + err.message + '). Pan/zoom to your site, then draw a rectangle.'
              );
            }
          },
          { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 }
        );
      } else if (hasFallback) {
        setMapCenter(fallbackLat, fallbackLon, 'Draw a <b>rectangle</b> over your site.');
      } else {
        setBanner('Geolocation not supported — pan/zoom to your site, then draw a rectangle.');
      }
    }

    function initPicker() {
      drawControl = new L.Control.Draw({
        draw: {
          polygon: false,
          polyline: false,
          circle: false,
          marker: false,
          circlemarker: false,
          rectangle: { shapeOptions: { color: '#2563eb' } }
        },
        edit: { featureGroup: drawn, edit: false, remove: true }
      });
      map.addControl(drawControl);

      map.on(L.Draw.Event.CREATED, function (e) {
        drawn.clearLayers();
        drawn.addLayer(e.layer);
        const bounds = e.layer.getBounds();
        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();
        const vertices = [
          [sw.lat, sw.lng],
          [sw.lat, ne.lng],
          [ne.lat, ne.lng],
          [ne.lat, sw.lng],
          [sw.lat, sw.lng]
        ];
        fetch('/roi', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ vertices: vertices })
        }).then(r => r.json()).then(data => {
          setBanner('<span style="color:#166534">' + data.message + '</span>');
          startStatusPoll();
        }).catch(err => {
          setBanner('<span style="color:#b91c1c">Error: ' + err + '</span>');
        });
      });
    }
    """

__all__ = ["ROIPickerError"]