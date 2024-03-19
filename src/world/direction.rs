use crate::prelude::*;

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Value)]
pub enum Direction {
    Null = 0,
    Left = 1,
    Right = 2,
    Down = 3,
    Up = 4,
    DownLeft = 5,
    DownRight = 6,
    UpLeft = 7,
    UpRight = 8,
}

impl Direction {
    pub fn iter_all() -> [Self; 9] {
        [
            Self::Null,
            Self::Left,
            Self::Right,
            Self::Down,
            Self::Up,
            Self::DownLeft,
            Self::DownRight,
            Self::UpLeft,
            Self::UpRight,
        ]
    }
    pub fn ortho(self) -> bool {
        self as u8 >= 1 && self as u8 <= 4
    }
    pub fn diag(self) -> bool {
        self as u8 >= 5
    }
    pub fn vector_table() -> [Vector2<i32>; 9] {
        [
            [0, 0],
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1],
        ]
        .map(|[x, y]| Vector2::new(x, y))
    }
    pub fn as_vector(self) -> Vector2<i32> {
        Self::vector_table()[self as usize]
    }
    pub fn as_vec(self) -> Vec2<i32> {
        self.as_vector().into()
    }
}
