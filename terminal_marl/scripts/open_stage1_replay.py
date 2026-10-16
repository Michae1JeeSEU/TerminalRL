"""Open a verified FT-RS archive in the existing local PyVista platform."""
import argparse
from ..core.rotating_slots.replay import ReplayBundle
from ..visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--time",type=float)
    parser.add_argument("--screenshots",metavar="DIRECTORY")
    parser.add_argument("--off-screen",action="store_true")
    args=parser.parse_args(argv)
    controller=ReplayController(ReplayBundle.load(args.bundle))
    if args.time is not None: controller.seek(args.time)
    viewer=RotatingSlotsReplayViewer(controller,off_screen=args.off_screen)
    try:
        if args.screenshots: print("\n".join(viewer.screenshots(args.screenshots)))
        if not args.off_screen: viewer.run()
    finally: viewer.close()

if __name__ == "__main__": main()
