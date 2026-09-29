"""Coordinates shared by native pinch handlers in nested Qt widgets."""


def native_position(widget, event):
    # QWidgetWindow forwards native gestures to child widgets without remapping
    # their window-local position (Qt 6.7.3, QTBUG-59595). Use screen coordinates
    # and the actual drawing widget instead. QPointF preserves subpixel input;
    # both positions are logical pixels, including on Retina displays.
    return widget.mapFromGlobal(event.globalPosition())
