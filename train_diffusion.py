# top-level shim so existing imports/tests keep working
from src.train_diffusion import *

if __name__ == "__main__":
    from src.train_diffusion import main
    main()
