use std::str::FromStr;

pub fn parse_str_to_args(s: &str) -> Vec<String> {
    let mut args = vec![];
    let mut acc = String::new();
    let mut has_quote = false;
    let mut ch = s.chars();
    while let Some(c) = ch.next() {
        if c.is_whitespace() && !has_quote {
            if !acc.is_empty() {
                has_quote = false;
                args.push(std::mem::replace(&mut acc, String::new()));
            }
            continue;
        }
        if c == '\"' {
            if !has_quote {
                has_quote = true;
            } else {
                has_quote = false;
            }
        } else if c == '\\' && has_quote {
            let n = ch
                .next()
                .expect("unexpected end of input when parsing arguments from string");
            let c = match n {
                '\\' => '\\',
                'n' => '\n',
                '\"' => '\"',
                '\'' => '\'',
                't' => '\t',
                _ => panic!("unrecognized escape character \\{}", n),
            };
            acc.push(c);
        } else {
            acc.push(c);
        }
    }
    if !acc.is_empty() {
        args.push(std::mem::replace(&mut acc, String::new()));
    }
    args
}
pub fn parse_arg<T: FromStr>(
    args: &[String],
    pos: &mut usize,
    name: &str,
    short: Option<&str>,
) -> Result<Option<T>, String> {
    assert!(name.starts_with("--"));
    assert!(short.is_none() || short.unwrap().starts_with("-"));
    let opt = &args[*pos];
    if !opt.starts_with("-") {
        return Err(format!(
            "expect form of `--options or --o` but found {}",
            opt
        ));
    }
    if opt == name || (short.is_some() && short.unwrap() == opt) {
        if *pos + 1 >= args.len() {
            return Err(format!("unexpected end of arguments when parsing {}", name));
        }
        let val = &args[*pos + 1];
        return match val.parse() {
            Err(_) => Err(format!(
                "error when parsing values of {}: val={}",
                name, val
            )),
            Ok(val) => {
                *pos += 2;
                Ok(Some(val))
            }
        };
    }
    Ok(None)
}
mod test {

    #[test]
    fn test_parse() {
        use super::parse_str_to_args;
        let a = "akr-cli -s cbox.json -a integrator.json";
        assert_eq!(
            parse_str_to_args(a),
            vec!["akr-cli", "-s", "cbox.json", "-a", "integrator.json"]
                .iter()
                .map(|x| String::from(*x))
                .collect::<Vec<_>>()
        );
        let b = r#"akr-cli -s "~/scene files/cbox.json" -a integrator.json"#;
        assert_eq!(
            parse_str_to_args(b),
            vec![
                "akr-cli",
                "-s",
                "~/scene files/cbox.json",
                "-a",
                "integrator.json"
            ]
            .iter()
            .map(|x| String::from(*x))
            .collect::<Vec<_>>()
        )
    }
}
