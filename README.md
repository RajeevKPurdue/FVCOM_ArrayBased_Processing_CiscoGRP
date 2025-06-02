## FVCOM_ArrayBased_Processing_CiscoGRP
### Rajeev Kumar, Höök Lab @ Purdue FNR; Collaboration with Mark Rowe (NOAA GLERL)
A streamlined, array based, alternative framework for processing FVCOM data that runs faster than PyFVCOM (but lacks high-level features) for those conducting big data analyses and do not have access to a computing cluster. I have also included R code for node based processing, spatial analyses, etc.. but R is prone to run into vector memory and speed issues on most PCs with Big Data.. even with the parallel, garbage collection, chunking, and full array based operations. 

The FVCOM data was provided by Mark Rowe (NOAA). The FVCOM model used to generate the output is described in (Rowe et al., 2019).

A volumetric analysis of Cisco habitat using GRP (growth rate potential) modeling (Brandt & Kirsch, 1992; Kitchell et al., 1977). Analyses of Lake Erie water quality and cold-water fish habitat for 17 historical years. Includes two Dissolved Oxygen Penalty frameworks (Arend et al., 2011; & a framework I may publish from work in my '.._IN_lakes..' repository- inspired by Jacobson et al., 2011).

The regex file pattern for reading in files is due to variable file size, dates and structure where the FVCOM can generate output files that contain 23 hours of one day and 1 hour of the next day. 

Assess your file structure and timescales of summary and modify the file processing and analysis scripts appropriately

## Attribution

## Citation

If you use this software in your research, please cite:

Kumar, R. (2025). FVCOM output array-based framework:  GitHub. https://github.com/RajeevKPurdue/FVCOM_ArrayBased_Processing_CiscoGRP


## License

This project is licensed under the [MIT](LICENSE) - see the LICENSE file for details.

## Acknowledgments

If you use this code in published work, please acknowledge its use and consider cite any relevant publications.
