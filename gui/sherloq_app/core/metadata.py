"""Pure conversion of ExifTool's grouped numeric JSON into display rows."""
import json

IGNORED_TAGS = ('SourceFile', 'ExifTool:ExifTool', 'File:FileName',
                'File:Directory', 'File:FileSize', 'File:FileModifyDate',
                'File:FileInodeChangeDate', 'File:FileAccessDate',
                'File:FileType', 'File:FilePermissions',
                'File:FileTypeExtension', 'File:MIMEType')


def metadata_rows(metadata):
    rows = []
    last = None
    for tag, value in metadata.items():
        # Zero/False are metadata values, not absent fields (e.g. sea-level GPS).
        if (not value and not isinstance(value, (int, float))) or any(t in tag for t in IGNORED_TAGS):
            continue
        value = str(value).replace(', use -b option to extract', '')
        value = value.replace('Binary data ', 'Binary data: ')
        group, separator, description = tag.partition(':')
        if not separator:
            group, description = 'Other', group
        rows.append([group if group != last else None, description, value])
        last = group
    return rows


def parse_metadata(data):
    documents = json.loads(data.decode('utf-8'))
    if not isinstance(documents, list) or len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError('Invalid metadata response from ExifTool.')
    return metadata_rows(documents[0])
