#[macro_export]
macro_rules! vector_union {
    (
        $name:ident, $type:ty, $fields_name:ident {
            $($field:ident $( : $nested_type:ident )?),*
        }
    ) => {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub union $name {
            pub values: [$type; vector_union!(@count $type, $($field $( : $nested_type)?),*)],
            pub fields: $fields_name,
        }

        impl $name {
            const SIZE: usize = vector_union!(@count $type, $($field $( : $nested_type)?),*);
        }

        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $fields_name {
            $(pub $field: vector_union!(@field_type $type, $($nested_type)?),)*
        }
    };

    (@count $type:ty, $($field:ident $( : $nested_type:ident )?),*) => {
        0 $(+ vector_union!(@field_size $type, $($nested_type)?))*
    };

    (@field_size $type:ty, $nested_type:ident) => {
        { $nested_type::SIZE }
    };

    (@field_size $type:ty,) => {
        1
    };

    (@field_type $type:ty, $nested_type:ident) => { $nested_type };
    (@field_type $type:ty,) => { $type };
}
