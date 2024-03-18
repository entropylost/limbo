pub struct LightFields {
    pub emission: VField<Vec3<f32>, Vec2<i32>>,
    // TODO: Make slices. Also figure out how to handle dispatching over them.
    pub light: VField<Vec3<f32>, Vec2<i32>>,
    // TODO: Figure out packing for this.
    pub wall: VField<bool, Vec2<i32>>,
    pub sunlight: VField<Vec3<f32>, u32>,
}
