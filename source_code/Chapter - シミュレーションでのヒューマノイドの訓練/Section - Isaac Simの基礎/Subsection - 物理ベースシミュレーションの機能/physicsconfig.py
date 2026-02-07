import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdPhysics, PhysxSchema

# シミュレーション取得
stage = omni.usd.get_context().get_stage()
scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())

# タイムステップ設定（2 ms, 4 substeps）
physx_scene_api.CreateTimeStepsPerSecondAttr().Set(500.0)
physx_scene_api.CreateMinPositionItersAttr().Set(4)
physx_scene_api.CreateMinVelocityItersAttr().Set(1)

# ソルバー反復回数
physx_scene_api.CreateSolverPositionIterationCountAttr().Set(50)
physx_scene_api.CreateSolverVelocityIterationCountAttr().Set(20)

# 接触オフセットとCCD
physx_scene_api.CreateContactOffsetAttr().Set(0.001)
physx_scene_api.CreateRestOffsetAttr().Set(0.0)
physx_scene_api.CreateEnableCCDAttr().Set(True)

# デフォルトマテリアル
material_path = "/World/defaultMaterial"
prim_utils.create_prim(material_path, "PhysicsMaterial")
material_prim = stage.GetPrimAtPath(material_path)
material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
material_api.CreateStaticFrictionAttr().Set(0.9)
material_api.CreateDynamicFrictionAttr().Set(0.9)
material_api.CreateRestitutionAttr().Set(0.02)

# GPU加速（利用可能な場合）
physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")