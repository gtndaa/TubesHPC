#include "../include/nbody.h"
#include "raylib.h"

#define SCREEN_WIDTH 1400
#define SCREEN_HEIGHT 900

#define SCALE 2.0f

static void draw_particles(Particle *p, int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = (float)(p[i].x * SCALE);
        float y = (float)(p[i].y * SCALE);

        if(i == 0)
        {
            DrawCircleV(
                (Vector2){x, y},
                10.0f,
                YELLOW
            );
        }
        else
        {
            DrawCircleV(
                (Vector2){x, y},
                2.0f,
                WHITE
            );
        }
    }
}

int main(int argc, char *argv[])
{
    SimParams params;
    parse_args(argc, argv, &params);

    int N = 1024;

    Particle *particles =
        (Particle*)malloc(N * sizeof(Particle));

    if (!particles)
    {
        printf("Allocation failed\n");
        return 1;
    }

    init_particles(
        particles,
        N,
        params.seed
    );

    InitWindow(
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        "N-Body Simulation (Raylib)"
    );

    SetTargetFPS(60);

    Camera2D camera = {0};

    camera.target = (Vector2){0.0f, 0.0f};
    camera.offset = (Vector2){
        SCREEN_WIDTH / 2.0f,
        SCREEN_HEIGHT / 2.0f
    };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    int step = 0;

    while (!WindowShouldClose())
    {
        /* ======================
           CAMERA CONTROL
           ====================== */

        float wheel = GetMouseWheelMove();

        if (wheel != 0)
        {
            Vector2 mouseWorld =
                GetScreenToWorld2D(
                    GetMousePosition(),
                    camera
                );

            camera.offset =
                GetMousePosition();

            camera.target =
                mouseWorld;

            camera.zoom += wheel * 0.1f;

            if (camera.zoom < 0.05f)
                camera.zoom = 0.05f;

            if (camera.zoom > 50.0f)
                camera.zoom = 50.0f;
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
        {
            Vector2 delta =
                GetMouseDelta();

            camera.target.x -=
                delta.x / camera.zoom;

            camera.target.y -=
                delta.y / camera.zoom;
        }

        if (IsKeyPressed(KEY_R))
        {
            camera.target =
                (Vector2){0, 0};

            camera.offset =
                (Vector2){
                    SCREEN_WIDTH / 2.0f,
                    SCREEN_HEIGHT / 2.0f
                };

            camera.zoom = 1.0f;
        }

        /* ======================
           PHYSICS UPDATE
           ====================== */

        compute_forces_serial(
            particles,
            N
        );

        velocity_verlet_step(
            particles,
            N,
            params.dt
        );

        step++;

        /* ======================
           DRAW
           ====================== */

        BeginDrawing();

        ClearBackground(BLACK);

        BeginMode2D(camera);

        draw_particles(
            particles,
            N
        );

        EndMode2D();

        DrawText(
            TextFormat("Particles: %d", N),
            20,
            20,
            20,
            GREEN
        );

        DrawText(
            TextFormat("Step: %d", step),
            20,
            50,
            20,
            GREEN
        );

        DrawText(
            TextFormat("Zoom: %.2f", camera.zoom),
            20,
            80,
            20,
            YELLOW
        );

        DrawText(
            "Mouse Wheel : Zoom",
            20,
            120,
            18,
            LIGHTGRAY
        );

        DrawText(
            "Right Drag : Pan",
            20,
            145,
            18,
            LIGHTGRAY
        );

        DrawText(
            "R : Reset Camera",
            20,
            170,
            18,
            LIGHTGRAY
        );

        DrawFPS(
            20,
            210
        );

        EndDrawing();
    }

    CloseWindow();
    free(particles);

    return 0;
}

