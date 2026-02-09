#!/usr/bin/env python
"""
Interactive play script for procgen environments.

Uses pygame for rendering and keyboard input.
"""
import argparse

import numpy as np

from .env import ENV_NAMES, ProcgenVecEnv


def main():
    default_str = "(default: %(default)s)"
    parser = argparse.ArgumentParser(
        description="Interactive version of Procgen allowing you to play the games"
    )
    parser.add_argument(
        "--vision",
        default="human",
        choices=["agent", "human"],
        help="level of fidelity of observation " + default_str,
    )
    parser.add_argument("--record-dir", help="directory to record movies to")
    parser.add_argument(
        "--distribution-mode",
        default="hard",
        help="which distribution mode to use for the level generation " + default_str,
    )
    parser.add_argument(
        "--env-name",
        default="coinrun",
        help="name of game to create " + default_str,
        choices=ENV_NAMES + ["coinrun_old"],
    )
    parser.add_argument(
        "--level-seed", type=int, help="select an individual level to use"
    )

    advanced_group = parser.add_argument_group("advanced optional switch arguments")
    advanced_group.add_argument(
        "--paint-vel-info",
        action="store_true",
        default=False,
        help="paint player velocity info in the top left corner",
    )
    advanced_group.add_argument(
        "--use-generated-assets",
        action="store_true",
        default=False,
        help="use randomly generated assets in place of human designed assets",
    )
    advanced_group.add_argument(
        "--uncenter-agent",
        action="store_true",
        default=False,
        help="display the full level for games that center the observation to the agent",
    )
    advanced_group.add_argument(
        "--disable-backgrounds",
        action="store_true",
        default=False,
        help="disable human designed backgrounds",
    )
    advanced_group.add_argument(
        "--restrict-themes",
        action="store_true",
        default=False,
        help="restricts games that use multiple themes to use a single theme",
    )
    advanced_group.add_argument(
        "--use-monochrome-assets",
        action="store_true",
        default=False,
        help="use monochromatic rectangles instead of human designed assets",
    )

    args = parser.parse_args()

    kwargs = {
        "paint_vel_info": args.paint_vel_info,
        "use_generated_assets": args.use_generated_assets,
        "center_agent": not args.uncenter_agent,
        "use_backgrounds": not args.disable_backgrounds,
        "restrict_themes": args.restrict_themes,
        "use_monochrome_assets": args.use_monochrome_assets,
        "render_mode": "rgb_array",
    }
    if args.env_name != "coinrun_old":
        kwargs["distribution_mode"] = args.distribution_mode
    if args.level_seed is not None:
        kwargs["start_level"] = args.level_seed
        kwargs["num_levels"] = 1

    env = ProcgenVecEnv(num_envs=1, env_name=args.env_name, **kwargs)

    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive mode: pip install pygame")
        return

    pygame.init()
    scale = 8
    width, height = 64 * scale, 64 * scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Procgen - {args.env_name}")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    running = True
    saved_state = None

    key_map = {
        pygame.K_LEFT: "LEFT",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_UP: "UP",
        pygame.K_DOWN: "DOWN",
        pygame.K_d: "D",
        pygame.K_a: "A",
        pygame.K_w: "W",
        pygame.K_s: "S",
        pygame.K_q: "Q",
        pygame.K_e: "E",
        pygame.K_LSHIFT: "LEFT_SHIFT",
        pygame.K_F1: "F1",
    }

    while running:
        keys_clicked = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    keys_clicked.add(key_map[event.key])

        pressed = pygame.key.get_pressed()
        keys_held = set()
        for pg_key, name in key_map.items():
            if pressed[pg_key]:
                keys_held.add(name)

        # State save/load
        if "LEFT_SHIFT" in keys_held and "F1" in keys_clicked:
            print("save state")
            saved_state = env.get_state()
        elif "F1" in keys_clicked:
            print("load state")
            if saved_state is not None:
                env.set_state(saved_state)

        actions = env.keys_to_act([keys_held])
        action = actions[0]
        if action is None:
            action = np.array([4])  # no-op

        obs, rew, terminated, truncated, info = env.step(action)

        # Render
        frame = env.render()
        if frame is not None:
            frame = frame[0]  # first env
        else:
            frame = obs[0]

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (width, height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(15)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
