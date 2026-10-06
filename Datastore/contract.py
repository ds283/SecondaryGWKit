"""
The tables and columns the datastore layer owns as its contract (prompts/datastore-generic, prompt
07; README §6.2, U4). Each is named once, here.

**Why these are the layer's, not the project's.** Every storable row carries a foreign key to the
version table's ``serial``, which ``Datastore/SQL/schema.py`` prepends; a pool is opened under a
version label, finds or writes its row before any actor may insert, and gives every actor its
serial. Every tagged class associates its rows with the tag table, through an association table
whose column names a tag's serial, and the check at open and the inventory recognise those
association tables by it. A project cannot choose other names for these without the layer
changing, so they are not declared per project: the layer names them, and a client registers a
factory under each name. The layer builds the objects of these tables through those factories,
never by importing a client's classes.

This module imports nothing outside the standard library, so that any module of the layer can
import it at module scope.
"""

# the version table: its name, and the column holding the label a pool is opened under
VERSION_TABLE = "version"
VERSION_LABEL = "label"

# the tag table: its name, the column holding a tag's label, and the column of an association
# table that names a tag's serial
TAG_TABLE = "store_tag"
TAG_LABEL = "label"
TAG_SERIAL = "tag_serial"
