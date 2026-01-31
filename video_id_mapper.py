import os

class VideoIdMapper:
    TVSUM_IDS = [
        "-esJrBWj2d8", "0tmA_C6XwfM", "37rzWOQsNIw", "3eYKfiOEJNs", "4wU_LUjG5Ic",
        "91IHQYk1IQM", "98MoyGZKHXc", "AwmHb44_ouw", "Bhxk-O1Y7Ho", "E11zDS9XGzg",
        "EE-bNr36nyA", "EYqVtI9YWJA", "GsAD1KT1xo8", "HT5vyqe0Xaw", "Hl-__g2gn_A",
        "J0nA4VgnoCo", "JKpqYvAdIsw", "JgHubY5Vw3Y", "LRw_obCPUt0", "NyBmCxDoHJU",
        "PJrm840pAUI", "RBCABdttQmI", "Se3oxnaPsz0", "VuWGsYPqAX8", "WG0MBPpPC6I",
        "WxtbjNsCQ8A", "XkqCExn6_Us", "XzYM3PfTM4w", "Yi4Ij2NM7U4", "_xMr-HKMfVA",
        "akI8YFjEmUw", "b626MiF1ew4", "byxOvuiIJV0", "cjibtmSLxQ4", "eQu1rNs0an0",
        "fWutDQy1nnY", "gzDbaEs1Rlg", "i3wAGJaaktw", "iVt07TCkFM0", "jcoYJXDG9sw",
        "kLxoNp-UchI", "oDXZc0tZe04", "qqR6AEXwxoQ", "sTEELN-vY30", "uGu_10sucQo",
        "vdmoEJ5YbrQ", "xmEERLqJ2kU", "xwqBXPGE9pQ", "xxdtq8mxegs", "z_6gVvQb2d0",
    ]

    @classmethod
    def get_canonical_id(cls, video_path):
        filename = os.path.basename(video_path)
        pure_id = filename.replace(".mp4", "").split()[0]
        try:
            index = cls.TVSUM_IDS.index(pure_id)
            return f"video_{index + 1}"
        except ValueError:
            return pure_id