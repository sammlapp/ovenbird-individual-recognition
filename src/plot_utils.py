from matplotlib import font_manager
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
# Path to font inside your repo
font_path = Path(__file__).parent / "Gill Sans Light Regular.otf"

# Register font with Matplotlib
font_manager.fontManager.addfont(str(font_path))

# Get the actual font name (important!)
prop = font_manager.FontProperties(fname=str(font_path))
DEFAULT_FONT_NAME = prop.get_name()
plt.rcParams["font.family"] = DEFAULT_FONT_NAME

from matplotlib import pyplot as plt
def figsize(w,h):
    plt.rcParams['figure.figsize']=[w,h]
figsize(6.5,3) #default figure size for papers
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


spring_palette = [
    "#A1C48D",
    "#97B2C9",
    "#E3DA99",
    "#9E90AC",
    "#E7AF85",
    "#A36B6F",
    # end of cycle
    "#586f4a",
    "#4d6274",
    "#8e8544",
    "#4d4952",
    "#8d5e3a",
    "#7F141D",
    # cycle
    "#729360",  # "#4E6E58",
    "#63819b",
    "#b7ad5c",
    "#655e6c",
    "#ba7b4a",
    "#80353B",
    # cycle
    "#d7e5cf",
    "#d3dee7",
    "#f2efd5",
    "#cec9d3",
    "#f4dccb",
    "#E6C8CA",
    # cycle
    "#495345",
    "#484f56",
    "#636047",
    "#3f3d41",
    "#625143",
    "#690C14",
    # greys
    "#8A888C",
    "#545355",
    "#79777B",
    "#E0E0E1",
    "#080808",
]


palette = sns.color_palette(spring_palette)
sns.set_palette(palette)