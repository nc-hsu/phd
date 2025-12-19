from openquake.hazardlib.imt import (
    PGA, SA, AvgSA, RSD595)

IMT_MAP = {
    "PGA": PGA,
    "SA": SA,
    "AvgSA[0,3]": AvgSA,
    "AvgSA[0,6]": AvgSA,
    "AvgSA": AvgSA,
    "RSD595": RSD595
}