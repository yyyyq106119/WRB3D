def test_inference_mri_only_runtime(tiny_model, paired_tensors, covariances):
    mri, _ = paired_tensors
    output = tiny_model.infer(mri, *covariances, num_steps=5)
    assert output["B_raw"].shape == mri.shape
    assert output["sampling_timesteps"] == [10, 8, 6, 4, 2, 0]

