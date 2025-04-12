# FVCOM_ArrayBased_Processing_CiscoGRP
An alternative for Processing FVCOM data that runs faster than PyFVCOM for those who do not have access to a computing cluster
The FVCOM data was provided by Mark Rowe (NOAA). The FVCOM model used to generate the output is described in (Rowe et al., 2019).

A volumetric analysis of Cisco habitat using GRP (growth rate potential) modeling. Analyses of Lake Erie of 17 historical years. 

The regex file pattern for reading in files is due to variable file size, dates and structure where the FVCOM can generate output files that contain 23 hours of one day and 1 hour of the next day. 
  Assess your file structure and timescales of summary and modify the file processing and analysis scripts appropriately
