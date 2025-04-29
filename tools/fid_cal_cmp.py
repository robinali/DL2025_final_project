from cleanfid import fid

# Set directories
real_dir = "../database/Test"
fake_dir = "./fake"

# Compute FID
fid_score = fid.compute_fid(real_dir, fake_dir)
print("FID score:", fid_score)

