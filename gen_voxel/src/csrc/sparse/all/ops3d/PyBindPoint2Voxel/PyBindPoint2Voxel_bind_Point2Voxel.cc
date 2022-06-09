// #include <csrc/sparse/all/ops3d/PyBindPoint2Voxel.h>
// #include <csrc/sparse/all/ops3d/Point2Voxel.h>

// namespace csrc {
// namespace sparse {
// namespace all {
// namespace ops3d {
// void PyBindPoint2Voxel::bind_Point2Voxel(const pybind11::module_& module)   {
  
//   pybind11::class_<csrc::sparse::all::ops3d::Point2Voxel> module_Point2Voxel(module, "Point2Voxel");
//   module_Point2Voxel.def_property_readonly("grid_size", &csrc::sparse::all::ops3d::Point2Voxel::get_grid_size, pybind11::return_value_policy::automatic);
//   module_Point2Voxel.def(pybind11::init<std::array<float, 3>, std::array<float, 6>, int, int, int>(), pybind11::arg("vsize_xyz"), pybind11::arg("coors_range_xyz"), pybind11::arg("num_point_features"), pybind11::arg("max_num_voxels"), pybind11::arg("max_num_points_per_voxel"));
//   module_Point2Voxel.def("point_to_voxel_hash", &csrc::sparse::all::ops3d::Point2Voxel::point_to_voxel_hash, pybind11::arg("points"), pybind11::arg("clear_voxels") = true, pybind11::arg("empty_mean") = false, pybind11::arg("stream_int") = 0, pybind11::return_value_policy::automatic);
//   module_Point2Voxel.def_static("point_to_voxel_hash_static", &csrc::sparse::all::ops3d::Point2Voxel::point_to_voxel_hash_static, pybind11::arg("points"), pybind11::arg("voxels"), pybind11::arg("indices"), pybind11::arg("num_per_voxel"), pybind11::arg("hashdata"), pybind11::arg("point_indice_data"), pybind11::arg("points_voxel_id"), pybind11::arg("vsize"), pybind11::arg("grid_size"), pybind11::arg("grid_stride"), pybind11::arg("coors_range"), pybind11::arg("clear_voxels") = true, pybind11::arg("empty_mean") = false, pybind11::arg("stream_int") = 0, pybind11::return_value_policy::automatic);
//   module_Point2Voxel.def_readonly("hashdata", &csrc::sparse::all::ops3d::Point2Voxel::hashdata);
//   module_Point2Voxel.def_readonly("point_indice_data", &csrc::sparse::all::ops3d::Point2Voxel::point_indice_data);
//   module_Point2Voxel.def_readonly("voxels", &csrc::sparse::all::ops3d::Point2Voxel::voxels);
//   module_Point2Voxel.def_readonly("indices", &csrc::sparse::all::ops3d::Point2Voxel::indices);
//   module_Point2Voxel.def_readonly("num_per_voxel", &csrc::sparse::all::ops3d::Point2Voxel::num_per_voxel);
// }
// } // namespace ops3d
// } // namespace all
// } // namespace sparse
// } // namespace csrc