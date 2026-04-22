from .flame_params_edit import FLAMEParamsEdit
from .flame_params_to_mesh import FLAMEParamsToMesh
from .freeuv_generate import FreeUVGenerate
from .freeuv_load import LoadFreeUV
from .kaolrm_load import LoadKaoLRM
from .kaolrm_preprocess import KaoLRMPreprocess
from .kaolrm_reconstruct import KaoLRMReconstruct
from .mesh_preview import MeshPreview
from .smirk_load import LoadSMIRK
from .smirk_predict import SMIRKPredict

NODE_CLASSES = [
    LoadKaoLRM,
    KaoLRMPreprocess,
    KaoLRMReconstruct,
    MeshPreview,
    LoadSMIRK,
    SMIRKPredict,
    FLAMEParamsEdit,
    FLAMEParamsToMesh,
    LoadFreeUV,
    FreeUVGenerate,
]
