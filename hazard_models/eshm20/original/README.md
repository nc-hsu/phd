<head>
<meta name="DC.title" content="European Seismic Hazard Model 2020 (ESHM20)">
<meta name="DC.bibliographicCitation" content="<![CDATA[The 2020 update of the European Seismic Hazard Model: Model Overview, https://doi.org/10.12686/a15, EFEHR Technical Report 001]]>">
<meta name="DC.type" content="Dataset">
<meta name="DC.publisher" content="EFEHR (European Facilities of Earthquake Hazard and Risk)">
<meta name="DC.identifier" scheme="DCTERMS.URI" content="https://doi.org/10.12686/a15">
<meta name="DC.format" content="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">
<meta name="DC.rights" scheme="DCTERMS.URI" content="info:eu-repo/semantics/openAccess">
<meta name="DC.FileFormat" content="application/x-shapefile">
<meta name="DC.FileFormat" content="application/xml">
<meta name="DC.rights.rightsHolder" content="EFEHR (European Facilities of Earthquake Hazard and Risk)">
<meta name="DC.license" content="<![CDATA[Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)]]>">
<meta name="DCTERMS.license" scheme="DCTERMS.URI" content="https://creativecommons.org/licenses/by/4.0/">
<link rel="cite-as" href="https://doi.org/10.12686/a15">
<link rel="describedby" href="https://data.crosscite.org/application/vnd.datacite.datacite+json/10.12686/a15" type="application/ld+json">
<link rel="describedby" href="https://data.crosscite.org/application/x-research-info-systems/10.12686/a15" type="application/x-research-info-systems">
<link rel="describedby" href="https://data.crosscite.org/application/x-bibtex/10.12686/a15" type="application/x-bibtex">
<link rel="item" href="https://doi.org/10.12686/eshm20-main-datasets" type="application/x-shapefile">
<link rel="item" href="https://doi.org/10.12686/eshm20-oq-input" type="application/xml">
<link rel="item" href="https://efehrappsrvr.ethz.ch/share/" type="application/xml">
<link rel="item" href="https://efehrmaps.ethz.ch/cgi-bin/mapserv?map=/var/www/mapfile/sharehazard.01.map&SERVICE=WMS&VERSION=1.3.&REQUEST=GetCapabilities" type="application/vnd.ogc.wms_xml">
</head>

# European Seismic Hazard Model 2020

[![Static Badge](https://img.shields.io/badge/DOI-10.12686%2Fa15-blue)](https://doi.org/10.12686/a15)
 [![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This is a public repository that contains data and resources including the 
final hazard models for the European Seismic Hazard Model 2020 (ESHM20).
This model covers the entire Euro-Mediterranean region, including Iceland, Turkey,
and the islands south and west of Spain.


Please use the following reference for citing this repository:

  *Danciu L., Nandan S., Reyes C., Basili R., Weatherill G., Beauval C., Rovida A., Vilanova S., Sesetyan K., Bard P-Y., Cotton F., Wiemer S., Giardini D. (2021) - The 2020 update of the European Seismic Hazard Model: Model Overview, <!--[![DOI](https://doi.org/10.12686/a15)](https://doi.org/10.12686/a15)-->https://doi.org/10.12686/a15, EFEHR Technical Report 001.*

> :exclamation: The use of any of these files without proper citation violates the CC-BY 4.0 license and is strictly prohibited :exclamation:






### Folder structure and file format

The top level folders are:

* [`documentation`](https://gitlab.seismo.ethz.ch/efehr/eshm20/-/tree/master/documentation?ref_type=heads) contains the technical report *The 2020 update of the European Seismic Hazard Model: Model Overview, <!--[![DOI](https://doi.org/10.12686/a15)](https://doi.org/10.12686/a15)-->https://doi.org/10.12686/a15, EFEHR Technical Report 001*. [![Static Badge](https://img.shields.io/badge/DOI-10.12686%2Fa15-blue)](https://doi.org/10.12686/a15)

* [`input_shapefiles`](https://doi.org/10.12686/eshm20-main-datasets) contains the ESRI shapefiles containing
data that was used to run the ESHM20 model [![Static Badge](https://img.shields.io/badge/DOI-10.12686%2Feshm20--main--datasets-blue)](https://doi.org/10.12686/eshm20-main-datasets)

* [`oq_computational`](https://doi.org/10.12686/eshm20-oq-input) contains the configuration files and 
openquake input nrml (.xml) files used to run the model [![Static Badge](https://img.shields.io/badge/DOI-10.12686%2Feshm20--oq--input-blue)](https://doi.org/10.12686/eshm20-oq-input)


* [`additional_materials`](#additional_materials) contains supplementary materials,
such as plots that visualize various model attributes, as well as comparison plots.

### Results and data access 
[![Static Badge](https://img.shields.io/badge/DOI-10.12686%2Feshm20--output-blue)](https://doi.org/10.12686/eshm20-output)

ESHM20 results are distributed online and publicly accessible via the EFEHR hazard web platform (hazard.efehr.org). The main results are: hazard maps, hazard curves, uniform hazard spectra and disaggregation of ground shaking hazard levels.

 The following web services are available for displaying and accessing these results: 
 * Hazard Maps  ([OGC WMS endpoint](https://efehrmaps.ethz.ch/cgi-bin/mapserv?map=/var/www/mapfile/sharehazard.01.map&SERVICE=WMS&VERSION=1.3.&REQUEST=GetCapabilities))
 * Hazard Curves ([RESTful API](https://efehrappsrvr.ethz.ch/share/)) 
 * Uniform Hazard Spectra ([RESTful API](https://efehrappsrvr.ethz.ch/share/))



---------------------------------------------------------------------------------------
### Disclaimer

The findings, comments, statements or recommendations expressed herein are exclusively
of the author(s) and do not necessarily reflect the views and policies of the institutions listed here: i.e. Swiss Seismological Service, ETH Zurich, Istituto
Nazionale di Geofisica e Vulcanologia (INGV), German Research Centre for Geociences (GFZ), Institut des Sciences de la Terre (ISTerre), Instituto Superior
Tecnico (IST), Bogazici University, Kandilli Observatory and Earthquake Research Institute, Department of Earthquake Engineering, the EFEHR Consortium
or the European Union. 

The authors of ESHM20 have tried to make the information in this product as accurate as possible. However, they do not guarantee that
the information herein is totally accurate or complete. Therefore, you should not solely rely on this information when making a commercial decision. 

Users of information provided herein assume all liability arising from such use. While undertaking to provide practical and accurate information, the authors assume no liability for, nor express or imply any warranty with regard to the information contained hereafter.

Licence: creativecommons cc-by 4.0
https://creativecommons.org/licenses/by/4.0/

You are free to:
Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
Adapt — remix, transform, and build upon the material for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
###### <strong><font color="red">Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.</font></strong>

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

##### Notices:
You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation .
No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

For more please refer to:
https://creativecommons.org/licenses/by/4.0/deed.en#ref-indicate-changes


**Contact Us**
If you have any questions or feedback on the data included in this repository, please send it via email to 'efehr.hazard@sed.ethz.ch'.


