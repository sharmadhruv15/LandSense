
class Config:
    BASE_PATH = '/content/landslide4sense/'
    
    TRAIN_IMG_DIR = BASE_PATH + 'TrainData/img/'
    TRAIN_MASK_DIR = BASE_PATH + 'TrainData/mask/'
    
    VALID_IMG_DIR = BASE_PATH + 'ValidData/img/'
    VALID_MASK_DIR = BASE_PATH + 'ValidData/mask/'
    
    TEST_IMG_DIR = BASE_PATH + 'TestData/img/'
    TEST_MASK_DIR = BASE_PATH + 'TestData/mask/'
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    NUM_CHANNELS = 14
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    SEED = 42
    
    # Model choice
    MODEL = "unet"          # Change to "mask_rcnn" later if needed

print("✅ Config file created!")
