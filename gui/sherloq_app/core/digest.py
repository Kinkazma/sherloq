"""Bounded, cancellable file digests. File bytes are freshly read for each run."""
import hashlib
import os
import re
from threading import Event
import cv2 as cv
import magic
from PySide6.QtCore import QFileInfo, QLocale, QCoreApplication
from gui.sherloq_app.core.utility import human_size

HASHES = (('MD5', 'md5'), ('SHA-1', 'sha1'), ('SHA2-224', 'sha224'),
          ('SHA2-256', 'sha256'), ('SHA2-384', 'sha384'), ('SHA2-512', 'sha512'),
          ('SHA3-224', 'sha3_224'), ('SHA3-256', 'sha3_256'),
          ('SHA3-384', 'sha3_384'), ('SHA3-512', 'sha3_512'))
IMAGE_HASHES = (('Average', cv.img_hash.averageHash),
               ('Block mean', cv.img_hash.blockMeanHash),
               ('Color moments', cv.img_hash.colorMomentHash),
               ('Marr-Hildreth', cv.img_hash.marrHildrethHash),
               ('Perceptual', cv.img_hash.pHash),
               ('Radial variance', cv.img_hash.radialVarianceHash))
CHUNK_BYTES = 1024 * 1024


def tr(text):
    return QCoreApplication.translate('DigestWidget', text)


def identity(stat):
    # Access time is intentionally excluded: this analysis itself reads the file.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns, stat.st_mode, stat.st_uid, stat.st_gid)


def ballistics(filename):
    table = [
        ["^DSCN[0-9]{4}\\.JPG$", "Nikon Coolpix camera"],
        ["^DSC_[0-9]{4}\\.JPG$", "Nikon digital camera"],
        ["^FUJI[0-9]{4}\\.JPG$", "Fujifilm digital camera"],
        ["^IMG_[0-9]{4}\\.JPG$", "Canon DSLR or iPhone camera"],
        ["^PIC[0-9]{5}\\.JPG$", "Olympus D-600L camera"],
    ]
    for entry in table:
        if re.match(entry[0], filename, re.IGNORECASE) is not None:
            return entry[1]
    return "Unknown source or manually renamed"


def physical_rows(filename):
    table = []

    file_info = QFileInfo(filename)
    table.append(
        [tr("PhysicalFile"), tr("File name"), file_info.fileName()]
    )
    table.append(
        [None, tr("Parent folder"), str(file_info.dir().absolutePath())]
    )
    table.append([None, tr("MIME type"), magic.from_file(filename, mime=True)])
    table.append(
        [
            None,
            tr("File size"),
            f"{QLocale().toString(file_info.size())} bytes ({human_size(file_info.size())})",
        ]
    )
    table.append([None, tr("File owner"), file_info.owner()])
    table.append(
        [None, tr("Permissions"), str(oct(os.stat(filename).st_mode)[-3:])]
    )
    table.append(
        [
            None,
            tr("Creation time"),
            file_info.birthTime().toLocalTime().toString(),
        ]
    )
    table.append(
        [
            None,
            tr("Last access"),
            file_info.lastRead().toLocalTime().toString(),
        ]
    )
    table.append(
        [
            None,
            tr("Last modified"),
            file_info.lastModified().toLocalTime().toString(),
        ]
    )
    table.append(
        [
            None,
            tr("Metadata changed"),
            file_info.metadataChangeTime().toLocalTime().toString(),
        ]
    )
    table.append(
        [None, tr("Name ballistics"), ballistics(file_info.fileName())]
    )
    return table


class DigestEngine:
    def __init__(self, filename, image):
        self.filename = filename
        self.image = image
        self.cancelled = Event()
        self.progress = 0

    def compute(self, _):
        self.progress = 0
        if self.cancelled.is_set():
            return None
        with open(self.filename, 'rb', buffering=0) as file:
            stamp = identity(os.fstat(file.fileno()))
            table = physical_rows(self.filename)
            states = [hashlib.new(name) for _, name in HASHES]
            buffer = bytearray(CHUNK_BYTES)
            view = memoryview(buffer)
            read = 0
            while not self.cancelled.is_set():
                count = file.readinto(buffer)
                if not count:
                    break
                for state in states:
                    state.update(view[:count])
                read += count
                self.progress = min(80, int(80 * read / max(1, stamp[2])))
            if self.cancelled.is_set():
                return None
            if identity(os.fstat(file.fileno())) != stamp:
                raise RuntimeError(tr('File changed during analysis. Please reload the image.'))
        for index, ((label, _), state) in enumerate(zip(HASHES, states)):
            table.append([tr('CryptoHash') if index == 0 else None, tr(label), state.hexdigest()])
        image_hashes = []
        for index, (label, calculate) in enumerate(IMAGE_HASHES):
            if self.cancelled.is_set():
                return None
            value = calculate(self.image)
            image_hashes.append(value)
            table.append([tr('ImageHash') if index == 0 else None, tr(label), str(value[0])])
            self.progress = 80 + int(20 * (index + 1) / len(IMAGE_HASHES))
        # Detect replacement of the pathname as well as writes to the opened file.
        if identity(os.stat(self.filename)) != stamp:
            raise RuntimeError(tr('File changed during analysis. Please reload the image.'))
        return table, image_hashes
