use super::prelude::*;

pub fn rainbow_map(mut h: f32) -> Color {
    let mut color = Color::default();
    color.w = 1.0;

    let s = 0.4;
    let v = 1.0;

    h -= h.floor();
    h *= 6.0;

    let i = h.floor() as i32;
    let mut f = h - i as f32;

    if i & 1 == 0 {
        f = 1.0 - f;
    }

    let m = v * (1.0 - s);
    let n = v * (1.0 - s * f);

    match i {
        6 | 0 => {
            color.x = v;
            color.y = n;
            color.z = m;
        }
        1 => {
            color.x = n;
            color.y = v;
            color.z = m;
        }
        2 => {
            color.x = m;
            color.y = v;
            color.z = n;
        }
        3 => {
            color.x = m;
            color.y = n;
            color.z = v;
        }
        4 => {
            color.x = n;
            color.y = m;
            color.z = v;
        }
        5 => {
            color.x = v;
            color.y = m;
            color.z = n;
        }
        _ => {
            color.x = 1.0;
            color.y = 0.5;
            color.z = 0.5;
        }
    }

    color
}

pub fn grayscale_map(h: f32) -> Color {
    let r = h;

    Color::new(r, r, r, 1.0)
}
