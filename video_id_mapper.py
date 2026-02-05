import os

class VideoIdMapper:
    TVSUM_IDS = [
        "AwmHb44_ouw", "98MoyGZKHXc", "J0nA4VgnoCo", "gzDbaEs1Rlg", "XzYM3PfTM4w",
        "HT5vyqe0Xaw", "sTEELN-vY30", "vdmoEJ5YbrQ", "xwqBXPGE9pQ", "akI8YFjEmUw",
        "i3wAGJaaktw", "Bhxk-O1Y7Ho", "0tmA_C6XwfM", "3eYKfiOEJNs", "xxdtq8mxegs",
        "WG0MBPpPC6I", "Hl-__g2gn_A", "Yi4Ij2NM7U4", "37rzWOQsNIw", "LRw_obCPUt0",
        "cjibtmSLxQ4", "b626MiF1ew4", "XkqCExn6_Us", "GsAD1KT1xo8", "PJrm840pAUI",
        "91IHQYk1IQM", "RBCABdttQmI", "z_6gVvQb2d0", "fWutDQy1nnY", "4wU_LUjG5Ic",
        "VuWGsYPqAX8", "JKpqYvAdIsw", "xmEERLqJ2kU", "byxOvuiIJV0", "_xMr-HKMfVA",
        "WxtbjNsCQ8A", "uGu_10sucQo", "EE-bNr36nyA", "Se3oxnaPsz0", "oDXZc0tZe04",
        "qqR6AEXwxoQ", "EYqVtI9YWJA", "eQu1rNs0an0", "JgHubY5Vw3Y", "iVt07TCkFM0",
        "E11zDS9XGzg", "NyBmCxDoHJU", "kLxoNp-UchI", "jcoYJXDG9sw", "-esJrBWj2d8"
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
    
    @classmethod
    def get_youtube_id(cls, alias_id):
        """convert alias id like video_1 to youtube id"""
        if alias_id.startswith("video_"):
            try:
                idx = int(alias_id.replace("video_", "")) - 1
                if 0 <= idx < len(cls.TVSUM_IDS):
                    return cls.TVSUM_IDS[idx]
            except ValueError:
                pass
        return alias_id