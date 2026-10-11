# Porto data attribution and reuse

The Porto Medium v0.2 release combines public taxi trajectories and an OpenStreetMap driving network. Preserve this notice and the dataset manifest when copying or redistributing it.

## Taxi trajectories

Source: Luis Moreira-Matias, Michel Ferreira, Joao Mendes-Moreira and the dataset contributors, **Taxi Service Trajectory – Prediction Challenge, ECML PKDD 2015**, UCI Machine Learning Repository, DOI [10.24432/C55W25](https://doi.org/10.24432/C55W25).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015) identifies the source license as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

This release changes the original data: it filters trips, recovers road routes, hashes trip identifiers, splits trips chronologically, and estimates candidate edge costs. It omits customer, taxi and call identifiers and does not redistribute the raw source CSV. Recovered road routes and preference costs are proxies, not independently verified route-choice labels.

## Road network and derived database

**© OpenStreetMap contributors.** The road network and the OSM-derived processed database are made available under the [Open Data Commons Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). See [OpenStreetMap copyright and attribution](https://www.openstreetmap.org/copyright).

This applies to the frozen GraphML, node/edge tables, topology, recovered road-route database, and edge-indexed cost arrays. The original taxi-source attribution and CC BY 4.0 notice remain applicable to the taxi-derived content. Keep both notices; the combined package does not erase either source's terms. Maps or figures using the road data should display OpenStreetMap attribution.

The benchmark processing was performed by the PWC project. The source providers do not endorse the benchmark or its algorithms. The data notice does not assign a new license to the repository's pre-existing research code.
