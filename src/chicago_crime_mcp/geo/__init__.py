"""Neighborhoods: the geography the city does not put on an incident.

Chicago tags every incident with a beat, district, ward and community area, but
not with a neighborhood -- and the neighborhood is what people actually ask
about. The 98 published neighborhood polygons are the sharper geography: Wicker
Park is 21% of West Town, so answering "robberies in Wicker Park" from the
community area covers roughly five times the ground that was asked about.

This package supplies both halves of closing that gap, and they are split by
what they need rather than by what they do:

- :mod:`~chicago_crime_mcp.geo.boundaries` does point-in-polygon tagging, needs
  DuckDB's spatial extension, and is imported **only by ingest**. The column is
  derived once, at ingest, and landed in Parquet -- the same shape as
  ``stable_category``, so the rule cannot drift between the two stores.
- :mod:`~chicago_crime_mcp.geo.resolve` turns a name a person typed into a value
  to filter on, holds **no geometry at all**, and is imported only by the server.

That split is deliberate. Because queries answer from the tagged column, the
serving path never links the spatial extension and never reaches the network
that ``INSTALL spatial`` needs on first use.

Two facts shape everything here. The neighborhood column is **nullable** --
about 1.8% of rows have no coordinates or fall outside every polygon -- while
``community_area`` is complete, so an answer has to say which geography it used.
And the set of 98 names is **inherently incomplete**: Bronzeville and Pilsen are
real places with no polygon, which is why fuzzy matching may suggest a
neighborhood but must never resolve one.
"""
