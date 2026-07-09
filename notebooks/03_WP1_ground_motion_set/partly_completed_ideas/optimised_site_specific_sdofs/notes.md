The notebooks in this folder use the cyclic pushover results from the site specific 3s CBF frames to determine an "optimised" equivalent SDOF system for each building and site.

This work would be useful if we wanted to extend or verify our equations for estimating the SDOF properties of the structure - or if we wanted to replace the current approximate SDOF systems with more building-specific ones.

The workflow is not finished.
1. There are several problematic pushover curves that did not run long enough to be able to calculate a brace contribution or there were problems with the envelope fitting. (site 15 and 38). The problem with the envelope fitting should be fixed as I have updated fitpo, however the updated envelope algorithm and options have not be added to these notebooks. Regarding the incomplete pushover - something needs to be done to get it to work...

2. The optimisation algorithm has not been run and I am unsure how time consuming it will be for all 60 sites. I could be several hours total... 